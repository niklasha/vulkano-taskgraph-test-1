use std::{slice, sync::Arc};
use std::time::Instant;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfoTyped};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::{
    physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
    QueueCreateInfo, QueueFlags,
};
use vulkano::memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator, GenericMemoryAllocatorCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::PipelineShaderStageCreateInfo;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout};
use vulkano::sync::{self, GpuFuture};
use vulkano::{DeviceSize, Version, VulkanLibrary};
use vulkano_taskgraph::{Id, Task};
use vulkano_taskgraph::QueueFamilyType;
use vulkano_taskgraph::resource::{AccessTypes, Resources, ResourcesCreateInfo};
use vulkano_taskgraph::graph::{CompileInfo, TaskGraph};
use half::f16;

// 1. Define the compute shaders (GLSL source embedded and compiled at build time)
mod cs_matmul {
    vulkano_shaders::shader!{                          // Use Vulkano’s shader! macro
        ty: "compute",                                  // It's a compute shader
        src: r#"
        #version 450
        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
        // Matrices: A is MxK, B is KxN, output C is MxN
        layout(push_constant) uniform Params { uint M; uint N; uint K; } params;
        layout(set = 0, binding = 0) readonly buffer A { float a[]; } bufA;
        layout(set = 0, binding = 1) readonly buffer B { float b[]; } bufB;
        layout(set = 0, binding = 2) writeonly buffer C { float c[]; } bufC;
        void main() {
            uint row = gl_GlobalInvocationID.x;
            uint col = gl_GlobalInvocationID.y;
            if(row < params.M && col < params.N) {
                float sum = 0.0;
                for(uint k = 0; k < params.K; ++k) {
                    sum += bufA.a[row * params.K + k] * bufB.b[k * params.N + col];
                }
                bufC.c[row * params.N + col] = sum;
            }
        }
        "#,
    }
}
mod cs_matmul_f16 {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r#"
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
        #extension GL_EXT_shader_16bit_storage : require
        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
        layout(push_constant) uniform Params { uint M; uint N; uint K; } params;
        layout(set = 0, binding = 0) readonly buffer A { float16_t a[]; } bufA;
        layout(set = 0, binding = 1) readonly buffer B { float16_t b[]; } bufB;
        layout(set = 0, binding = 2) writeonly buffer C { float16_t c[]; } bufC;
        void main() {
            uint row = gl_GlobalInvocationID.x;
            uint col = gl_GlobalInvocationID.y;
            if (row < params.M && col < params.N) {
                float sum = 0.0;
                for (uint k = 0; k < params.K; ++k) {
                    sum += float(bufA.a[row * params.K + k]) * float(bufB.b[k * params.N + col]);
                }
                sum = clamp(sum, -65504.0, 65504.0);
                bufC.c[row * params.N + col] = float16_t(sum);
            }
        }
        "#,
    }
}
mod cs_activate {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r#"
        #version 450
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        layout(push_constant) uniform Params { uint count; } params; // number of elements
        layout(set = 0, binding = 0) readonly buffer In { float x[]; } bufIn;
        layout(set = 0, binding = 1) writeonly buffer Out { float y[]; } bufOut;
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if(idx < params.count) {
                float val = bufIn.x[idx];
                // Example activation: ReLU (threshold at 0)
                bufOut.y[idx] = (val > 0.0) ? val : 0.0;
            }
        }
        "#,
    }
}
mod cs_activate_f16 {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r#"
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
        #extension GL_EXT_shader_16bit_storage : require
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        layout(push_constant) uniform Params { uint count; } params;
        layout(set = 0, binding = 0) readonly buffer In { float16_t x[]; } bufIn;
        layout(set = 0, binding = 1) writeonly buffer Out { float16_t y[]; } bufOut;
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if(idx < params.count) {
                float16_t val = bufIn.x[idx];
                bufOut.y[idx] = (val > float16_t(0.0)) ? val : float16_t(0.0);
            }
        }
        "#,
    }
}
mod cs_reduce {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r#"
        #version 450
        // For simplicity, use a single workgroup for reduction
        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
        layout(push_constant) uniform Params { uint count; } params;
        layout(set = 0, binding = 0) readonly buffer In { float data[]; } bufIn;
        layout(set = 0, binding = 1) writeonly buffer Out { float result[]; } bufOut;
        shared float partial_sum[256];  // shared memory for intra-group reduction
        void main() {
        //            uint idx = gl_LocalInvocationID.x;
        //            uint global_idx = gl_GlobalInvocationID.x;
        //            // Load data into shared memory
        //            float value = 0.0;
        //            if(global_idx < params.count) {
        //                value = bufIn.data[global_idx];
        //            }
        //            partial_sum[idx] = value;
        //            memoryBarrierShared();
        //            barrier(); 
        //            // Reduction in shared memory (simple sequential reduction for demo)
        //            if(idx == 0) {
        //                float sum = 0.0;
        //                for(uint i = 0; i < 256; ++i) {
        //                    sum += partial_sum[i];
        //                }
        //                bufOut.result[0] += sum;
        //            }
              uint gid = gl_GlobalInvocationID.x;
              // Let a single thread do the whole reduction
              if (gid == 0) {
                  float sum = 0.0;
                  for (uint i = 0; i < params.count; ++i)
                      sum += bufIn.data[i];
                  bufOut.result[0] = sum;
              }
        }
        "#,
    }
}
mod cs_reduce_f16 {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r#"
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
        #extension GL_EXT_shader_16bit_storage : require
        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
        layout(push_constant) uniform Params { uint count; } params;
        layout(set = 0, binding = 0) readonly buffer In { float16_t data[]; } bufIn;
        layout(set = 0, binding = 1) writeonly buffer Out { float16_t result[]; } bufOut;
        void main() {
              uint gid = gl_GlobalInvocationID.x;
              if (gid == 0) {
                  float sum = 0.0;
                  for (uint i = 0; i < params.count; ++i)
                      sum += log(float(bufIn.data[i]) + 1.0);
                  float mean = sum / float(params.count);
                  bufOut.result[0] = float16_t(mean);
              }
        }
        "#,
    }
}

// 2. Define the World context carrying pipelines and descriptor sets
struct MyWorld {
    // Pipelines:
    matmul_pipeline: std::sync::Arc<ComputePipeline>,
    activate_pipeline: std::sync::Arc<ComputePipeline>,
    reduce_pipeline: std::sync::Arc<ComputePipeline>,
    // Pre-built descriptor sets for each pipeline (binding the buffers)
    matmul_desc_set: std::sync::Arc<DescriptorSet>,
    activate_desc_set: std::sync::Arc<DescriptorSet>,
    reduce_desc_set: std::sync::Arc<DescriptorSet>,
    // Push constant data:
    dims: cs_matmul::Params,      // struct { M, N, K }
    vec_len: u32,                    // count of elements for activation/reduction
    use_fp16: bool,
    use_wmma: bool,
}

// 3. Define Task implementations for each stage
struct MatMulTask;
impl Task for MatMulTask {
    type World = MyWorld;
    unsafe fn execute(
        &self,
        cb: &mut vulkano_taskgraph::command_buffer::RecordingCommandBuffer,
        _tcx: &mut vulkano_taskgraph::TaskContext,
        world: &MyWorld
    ) -> vulkano_taskgraph::TaskResult {
        // Bind pipeline and descriptor set, push constants, then dispatch
        cb.bind_pipeline_compute(&world.matmul_pipeline.clone()).unwrap();                  // bind compute pipeline
	cb.as_raw().bind_descriptor_sets(
            PipelineBindPoint::Compute,
            world.matmul_pipeline.layout(),
            0,
            &[world.matmul_desc_set.as_raw()],
            &[]
        ).unwrap();
        if world.use_fp16 {
            let pc = cs_matmul_f16::Params { M: world.dims.M, N: world.dims.N, K: world.dims.K };
            cb.push_constants(&world.matmul_pipeline.layout().clone(), 0, &pc).unwrap();
        } else {
            cb.push_constants(&world.matmul_pipeline.layout().clone(), 0, &world.dims).unwrap();
        }
        // Determine dispatch size: one thread per output element (M x N threads, grouped in 16x16)
        let (wg_x, wg_y) = ((world.dims.M + 15) / 16, (world.dims.N + 15) / 16);
        cb.dispatch([wg_x, wg_y, 1]).unwrap();
        Ok(())
    }
}
struct ActivationTask;
impl Task for ActivationTask {
    type World = MyWorld;
    unsafe fn execute(
        &self,
        cb: &mut vulkano_taskgraph::command_buffer::RecordingCommandBuffer,
        _tcx: &mut vulkano_taskgraph::TaskContext,
        world: &MyWorld
    ) -> vulkano_taskgraph::TaskResult {
        cb.bind_pipeline_compute(&world.activate_pipeline.clone()).unwrap();
	cb.as_raw().bind_descriptor_sets(
            PipelineBindPoint::Compute,
            world.activate_pipeline.layout(),
            0,
            &[world.activate_desc_set.as_raw()],
            &[]
        ).unwrap();
        let count = world.vec_len;
        if world.use_fp16 {
            cb.push_constants(&world.activate_pipeline.layout().clone(), 0, &cs_activate_f16::Params { count }).unwrap();
        } else {
            cb.push_constants(&world.activate_pipeline.layout().clone(), 0, &cs_activate::Params { count }).unwrap();
        }
        // Dispatch enough threads for 'count' elements
        let wg_count = (count + 63) / 64;
        cb.dispatch([wg_count, 1, 1]).unwrap();
        Ok(())
    }
}
struct ReduceTask;
impl Task for ReduceTask {
    type World = MyWorld;
    unsafe fn execute(
        &self,
        cb: &mut vulkano_taskgraph::command_buffer::RecordingCommandBuffer,
        _tcx: &mut vulkano_taskgraph::TaskContext,
        world: &MyWorld
    ) -> vulkano_taskgraph::TaskResult {
        cb.bind_pipeline_compute(&world.reduce_pipeline.clone()).unwrap();
	cb.as_raw().bind_descriptor_sets(
            PipelineBindPoint::Compute,
            world.reduce_pipeline.layout(),
            0,
            &[world.reduce_desc_set.as_raw()],
            &[]
        ).unwrap();
        let count = world.vec_len;
        if world.use_fp16 {
            cb.push_constants(&world.reduce_pipeline.layout().clone(), 0, &cs_reduce_f16::Params { count }).unwrap();
        } else {
            cb.push_constants(&world.reduce_pipeline.layout().clone(), 0, &cs_reduce::Params { count }).unwrap();
        }
        // Dispatch (we use one workgroup of 256 threads; ensure enough threads to cover 'count')
        let _wg_count = (count + 255) / 256;  // number of groups needed (likely 1 here)
        cb.dispatch([/*_wg_count*/1, 1, 1]).unwrap();
        Ok(())
    }
}

fn main() {
    // 4. Initialize Vulkan (instance, device, queue)
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        &library,
        &InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
    //        enabled_extensions: &required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    // ... select physical device and queue family that supports compute ...

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            p.api_version() >= Version::V1_1 || p.supported_extensions().khr_maintenance2
        })
//        .filter(|p| {
//            p.supported_extensions().contains(&device_extensions)
//                && p.supported_features().contains(&device_features)
//        })
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(_i, q)| {
                    q.queue_flags.contains(QueueFlags::COMPUTE)
//                        && p.presentation_support(i as u32, event_loop).unwrap()
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

    println!(
        "Using GPU: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    // Check for NVIDIA cooperative matrix (WMMA) support
    let wmma_supported = physical_device
        .supported_extensions()
        .nv_cooperative_matrix
        || physical_device.supported_extensions().khr_cooperative_matrix;

    if wmma_supported {
        println!("WMMA/cooperative matrix supported");
    } else {
        println!("WMMA/cooperative matrix NOT supported");
    }

    let use_fp16 = physical_device
        .supported_extensions()
        .khr_16bit_storage
        && physical_device.supported_features().shader_float16
        || wmma_supported;

    let device_extensions = DeviceExtensions {
        ext_shader_atomic_float: true,
        khr_16bit_storage: use_fp16,
        nv_cooperative_matrix: wmma_supported,
        khr_cooperative_matrix: wmma_supported,
        ..DeviceExtensions::empty()
    };

    let device_features = DeviceFeatures {
        storage_buffer16_bit_access: use_fp16,
        shader_float16: use_fp16,
        cooperative_matrix: wmma_supported,
        ..DeviceFeatures::empty()
    };

    let (device, mut queues) = Device::new(
        &physical_device,
        &DeviceCreateInfo {
            enabled_extensions: &device_extensions,
            enabled_features: &device_features,
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();
    let queue = queues.next().unwrap();
    let mem_alloc = Arc::new(StandardMemoryAllocator::new(
        &device.clone(),
        &GenericMemoryAllocatorCreateInfo::default(),
    ));
    let cmd_alloc = Arc::new(StandardCommandBufferAllocator::new(
        &device.clone(),
        &StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    // 5. Create Resources and a Flight
    let resources = Resources::new(
        &device,
        &ResourcesCreateInfo {
//            bindless_context: Some(&Default::default()),
            ..Default::default()
        },
    )
    .unwrap();
    let flight_id = resources.create_flight(1).unwrap();  // single-frame flight

    // 6. Allocate buffers for A, B, C, D, and result
    // Assume dimensions:
    // Increase matrix dimensions tenfold to better warm up the GPU
    let m=4096u32; let k=2048u32; let n=1024u32;
    let len_a = m * k;
    let len_b = k * n;
    let len_c = m * n;
    // Create GPU buffers (device local) via resources
    let elem_size = if use_fp16 { std::mem::size_of::<f16>() } else { std::mem::size_of::<f32>() } as DeviceSize;
    let a_id: Id<Buffer> = resources.create_buffer(
        &vulkano::buffer::BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
        DeviceLayout::from_size_alignment(
            len_a as DeviceSize * elem_size,
            if use_fp16 { std::mem::align_of::<f16>() as DeviceSize } else { std::mem::align_of::<f32>() as DeviceSize })
            .unwrap(),
    ).unwrap();
    let b_id: Id<Buffer> = resources.create_buffer(
        &vulkano::buffer::BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
        DeviceLayout::from_size_alignment(
            len_b as DeviceSize * elem_size,
            if use_fp16 { std::mem::align_of::<f16>() as DeviceSize } else { std::mem::align_of::<f32>() as DeviceSize })
            .unwrap(),
    ).unwrap();
    let c_id: Id<Buffer> = resources.create_buffer(
        &vulkano::buffer::BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),  // default allocation (device local)
        DeviceLayout::from_size_alignment(
            len_c as DeviceSize * elem_size,
            if use_fp16 { std::mem::align_of::<f16>() as DeviceSize } else { std::mem::align_of::<f32>() as DeviceSize })
            .unwrap(),
    ).unwrap();
    let d_id: Id<Buffer> = resources.create_buffer(
        &vulkano::buffer::BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),  // default allocation (device local)
        DeviceLayout::from_size_alignment(
            len_c as DeviceSize * elem_size,
            if use_fp16 { std::mem::align_of::<f16>() as DeviceSize } else { std::mem::align_of::<f32>() as DeviceSize })
            .unwrap(),
    ).unwrap();
    let r_id = resources.create_buffer(
        &vulkano::buffer::BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter:
                  MemoryTypeFilter::PREFER_HOST        // prefer host-visible heap
                | MemoryTypeFilter::HOST_RANDOM_ACCESS, // host read/write
            ..Default::default()
        },
        DeviceLayout::from_size_alignment(
            elem_size,
            if use_fp16 { std::mem::align_of::<f16>() as DeviceSize } else { std::mem::align_of::<f32>() as DeviceSize })
            .unwrap(),
    ).unwrap();
    // Fill host-visible portions of A and B with data (for demo, skip actual data fill)
    // In practice, use resources.write_buffer or the Buffer's mapped memory to initialize.
    // create  staging buffer with sequential floats 0,1,2,…
    let host_a_f16: Option<Subbuffer<[f16]>>;
    let host_a_f32: Option<Subbuffer<[f32]>>;
    if use_fp16 {
        let buf = Buffer::from_iter(
            &mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter:
                    MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_HOST,
                ..Default::default()
            },
            (0..len_a).map(|i| f16::from_f32(i as f32)),
        ).unwrap();
        host_a_f16 = Some(buf);
        host_a_f32 = None;
    } else {
        let buf = Buffer::from_iter(
            &mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter:
                    MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_HOST,
                ..Default::default()
            },
            (0..len_a).map(|i| i as f32),
        ).unwrap();
        host_a_f32 = Some(buf);
        host_a_f16 = None;
    }
    let host_b_f16: Option<Subbuffer<[f16]>>;
    let host_b_f32: Option<Subbuffer<[f32]>>;
    if use_fp16 {
        let buf = Buffer::from_iter(
            &mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter:
                    MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_HOST,
                ..Default::default()
            },
            (0..len_b).map(|i| f16::from_f32(i as f32)),
        ).unwrap();
        host_b_f16 = Some(buf);
        host_b_f32 = None;
    } else {
        let buf = Buffer::from_iter(
            &mem_alloc,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter:
                    MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_HOST,
                ..Default::default()
            },
            (0..len_b).map(|i| i as f32),
        ).unwrap();
        host_b_f32 = Some(buf);
        host_b_f16 = None;
    }

    // command-buffer copy host_a → a_id  (do once before executing the graph)
    let mut cb = AutoCommandBufferBuilder::primary(
        cmd_alloc.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap();
    if use_fp16 {
        let a_slice: Subbuffer<[f16]> = Subbuffer::from(resources.buffer(a_id).unwrap().buffer().clone()).reinterpret();
        let host = host_a_f16.as_ref().unwrap();
        cb.copy_buffer(CopyBufferInfoTyped::new(host.clone(), a_slice.clone())).unwrap();
    } else {
        let a_slice: Subbuffer<[f32]> = Subbuffer::from(resources.buffer(a_id).unwrap().buffer().clone()).reinterpret();
        let host = host_a_f32.as_ref().unwrap();
        cb.copy_buffer(CopyBufferInfoTyped::new(host.clone(), a_slice.clone())).unwrap();
    }
    let cb = cb.build().unwrap();
    let future = sync::now(device.clone())            // start with an idle future
        .then_execute(queue.clone(), cb).unwrap()             // submit to the queue
        .then_signal_fence_and_flush().unwrap();              // fence + flush
    future.wait(None).unwrap();                               // block until GPU finished

    // command-buffer copy host_b → b_id  (do once before executing the graph)
    let mut cb = AutoCommandBufferBuilder::primary(
        cmd_alloc.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap();
    if use_fp16 {
        let b_slice: Subbuffer<[f16]> = Subbuffer::from(resources.buffer(b_id).unwrap().buffer().clone()).reinterpret();
        let host = host_b_f16.as_ref().unwrap();
        cb.copy_buffer(CopyBufferInfoTyped::new(host.clone(), b_slice.clone())).unwrap();
    } else {
        let b_slice: Subbuffer<[f32]> = Subbuffer::from(resources.buffer(b_id).unwrap().buffer().clone()).reinterpret();
        let host = host_b_f32.as_ref().unwrap();
        cb.copy_buffer(CopyBufferInfoTyped::new(host.clone(), b_slice.clone())).unwrap();
    }
    let cb = cb.build().unwrap();
    let future = sync::now(device.clone())            // start with an idle future
        .then_execute(queue.clone(), cb).unwrap()             // submit to the queue
        .then_signal_fence_and_flush().unwrap();              // fence + flush
    future.wait(None).unwrap();                               // block until GPU finished

    // 7. Build compute pipelines and descriptor sets for each shader
    // (In a real program, pipeline creation and descriptor set writing would be done once at init)
    let shader_matmul = if use_fp16 {
        cs_matmul_f16::load(&device).unwrap().entry_point("main").unwrap()
    } else {
        cs_matmul::load(&device).unwrap().entry_point("main").unwrap()
    };
    let stage_matmul = PipelineShaderStageCreateInfo::new(&shader_matmul);
    let layout_matmul = PipelineLayout::from_stages(&device, slice::from_ref(&stage_matmul)).unwrap();
    let pipeline_matmul = ComputePipeline::new(
        &device,
        None,
        &ComputePipelineCreateInfo::new(stage_matmul, &layout_matmul),
    )
    .unwrap();

    let shader_activate = if use_fp16 {
        cs_activate_f16::load(&device).unwrap().entry_point("main").unwrap()
    } else {
        cs_activate::load(&device).unwrap().entry_point("main").unwrap()
    };
    let stage_activate = PipelineShaderStageCreateInfo::new(&shader_activate);
    let layout_activate = PipelineLayout::from_stages(&device, slice::from_ref(&stage_activate)).unwrap();
    let pipeline_activate = ComputePipeline::new(
        &device,
        None,
        &ComputePipelineCreateInfo::new(stage_activate, &layout_activate),
    )
    .unwrap();

    let shader_reduce = if use_fp16 {
        cs_reduce_f16::load(&device).unwrap().entry_point("main").unwrap()
    } else {
        cs_reduce::load(&device).unwrap().entry_point("main").unwrap()
    };
    let stage_reduce = PipelineShaderStageCreateInfo::new(&shader_reduce);
    let layout_reduce = PipelineLayout::from_stages(&device, slice::from_ref(&stage_reduce)).unwrap();
    let pipeline_reduce = ComputePipeline::new(
        &device,
        None,
        &ComputePipelineCreateInfo::new(stage_reduce, &layout_reduce),
    )
    .unwrap();
    // Create descriptor sets for each pipeline:
    use vulkano::descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        WriteDescriptorSet,
    };
    // We need to get the actual buffer objects from the resource IDs to bind them.
    let binding_a = resources.buffer(a_id).unwrap();
    let buf_a = binding_a.buffer();  // get Arc<Buffer>
    let binding_b = resources.buffer(b_id).unwrap();
    let buf_b = binding_b.buffer();  // get Arc<Buffer>
    let binding_c = resources.buffer(c_id).unwrap();
    let buf_c = binding_c.buffer();  // get Arc<Buffer>
    let binding_d = resources.buffer(d_id).unwrap();
    let buf_d = binding_d.buffer();  // get Arc<Buffer>
    let binding_r = resources.buffer(r_id).unwrap();
    let buf_r = binding_r.buffer();  // get Arc<Buffer>
    let r_slice_f16: Option<Subbuffer<[f16]>>;
    let r_slice_f32: Option<Subbuffer<[f32]>>;
    if use_fp16 {
        let slice: Subbuffer<[f16]> = Subbuffer::from(buf_r.clone()).reinterpret();
        {
            let mut guard = slice.write().unwrap();
            guard[0] = f16::from_f32(0.0);
        }
        r_slice_f16 = Some(slice);
        r_slice_f32 = None;
    } else {
        let slice: Subbuffer<[f32]> = Subbuffer::from(buf_r.clone()).reinterpret();
        {
            let mut guard = slice.write().unwrap();
            guard[0] = 0.0_f32;
        }
        r_slice_f32 = Some(slice);
        r_slice_f16 = None;
    }
    // Now allocate descriptor sets:
    let set_alloc = Arc::new(StandardDescriptorSetAllocator::new(
        &device.clone(),
        &Default::default(),   // StandardDescriptorSetAllocatorCreateInfo
    ));
    let matmul_set = DescriptorSet::new(
        set_alloc.clone(),
        pipeline_matmul.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, buf_a.clone().into()),  // binding 0 -> A
            WriteDescriptorSet::buffer(1, buf_b.clone().into()),  // binding 1 -> B
            WriteDescriptorSet::buffer(2, buf_c.clone().into()),  // binding 2 -> C output
        ],
        [],
    )
    .unwrap();
    let activate_set = DescriptorSet::new(
        set_alloc.clone(),
        pipeline_activate.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, buf_c.clone().into()),  // binding 0 -> input C
            WriteDescriptorSet::buffer(1, buf_d.clone().into()),  // binding 1 -> output D
        ],
        [],
    ).unwrap();
    let reduce_set = DescriptorSet::new(
        set_alloc.clone(),
        pipeline_reduce.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, buf_d.clone().into()),  // binding 0 -> input D
            WriteDescriptorSet::buffer(1, buf_r.clone().into()),  // binding 1 -> output scalar
        ],
        [],
    ).unwrap();

    // 8. Create and populate the TaskGraph
    let mut task_graph = TaskGraph::<MyWorld>::new(&resources);

    let virt_a = task_graph.add_buffer(&BufferCreateInfo::default());
    let virt_b = task_graph.add_buffer(&BufferCreateInfo::default());
    let virt_c = task_graph.add_buffer(&BufferCreateInfo::default());
    let virt_d = task_graph.add_buffer(&BufferCreateInfo::default());
    let virt_r = task_graph.add_buffer(&BufferCreateInfo::default());

    // Add each task to the graph:
    let mut node_mul = task_graph.create_task_node("MatrixMultiply", QueueFamilyType::Compute, MatMulTask);
    let node_mul = node_mul.buffer_access(virt_a, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
           .buffer_access(virt_b, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
           .buffer_access(virt_c, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
           .build();
    let mut node_act = task_graph.create_task_node("Activation", QueueFamilyType::Compute, ActivationTask);
    let node_act = node_act.buffer_access(virt_c, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
           .buffer_access(virt_d, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
           .build();
    let mut node_red = task_graph.create_task_node("Reduction", QueueFamilyType::Compute, ReduceTask);
    let node_red = node_red.buffer_access(virt_d, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
           .buffer_access(virt_r, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
           .build();
    // Connect dependencies: MatMul -> Activation -> Reduction
    task_graph.add_edge(node_mul, node_act).unwrap();
    task_graph.add_edge(node_act, node_red).unwrap();
    // Now the graph is set up (3 nodes, properly ordered)

    // 9. Compile the graph
    let compile_info = CompileInfo {
        queues: &[&queue],       // use our single compute queue for all tasks
        present_queue: None,     // no presentation needed
        flight_id,
        ..Default::default()
    };
    let exec_graph = unsafe { task_graph.compile(&compile_info) }
        .expect("Failed to compile task graph");

    // Prepare the World context with pipelines, descriptor sets, and parameters
    let world = MyWorld {
        matmul_pipeline: pipeline_matmul.clone(),
        activate_pipeline: pipeline_activate.clone(),
        reduce_pipeline: pipeline_reduce.clone(),
        matmul_desc_set: matmul_set.clone(),
        activate_desc_set: activate_set.clone(),
        reduce_desc_set: reduce_set.clone(),
        dims: cs_matmul::Params { M: m, N: n, K: k },
        vec_len: (m * n),      // total elements for activation and reduction
        use_fp16,
        use_wmma: wmma_supported,
    };

    // 10. Execute the task graph
    use vulkano_taskgraph::resource_map;
    // Create a ResourceMap mapping virtual resources to physical (here 1:1 mapping)
    let resource_map = resource_map! {
        &exec_graph, virt_a => a_id, virt_b => b_id, virt_c => c_id, virt_d => d_id, virt_r => r_id   // this macro maps each listed id to itself in the given Resources
    }
    .unwrap();

    let start_gpu = Instant::now();
    unsafe { exec_graph.execute(resource_map, &world, || {}) }.unwrap();
    unsafe { device.wait_idle() }.unwrap();
    let gpu_time = start_gpu.elapsed();
    // 11. Retrieve and print result
    // Read the result buffer:
    if use_fp16 {
        let slice = r_slice_f16.as_ref().unwrap();
        let result_guard = slice.read().unwrap();
        let result_data: &[f16] = &result_guard;
        println!("Mean log1p of all elements in C = {}", f32::from(result_data[0]));
    } else {
        let slice = r_slice_f32.as_ref().unwrap();
        let result_guard = slice.read().unwrap();
        let result_data: &[f32] = &result_guard;
        println!("Sum of all elements in C = {}", result_data[0]);
    }

    let start_cpu = Instant::now();
    let mut cpu_c = vec![0f32; (m * n) as usize];
    for row in 0..m {
        for col in 0..n {
            let mut s = 0f32;
            for kk in 0..k {
                s += (row*k + kk) as f32 * (kk*n + col) as f32
            }
            cpu_c[(row*n + col) as usize] = s.max(0.0);        // ReLU
        }
    }
    let golden: f32 = if use_fp16 {
        cpu_c
            .iter()
            .map(|x| {
                let clamped = x.clamp(-65504.0, 65504.0);
                (clamped + 1.0).ln()
            })
            .sum::<f32>()
            / (m * n) as f32
    } else {
        cpu_c.iter().copied().sum()
    };
    let cpu_time = start_cpu.elapsed();
    if use_fp16 {
        println!("CPU reference mean log1p = {}", golden);
    } else {
        println!("CPU reference sum = {}", golden);
    }
    println!("GPU time: {:?}, CPU time: {:?}", gpu_time, cpu_time);
}
