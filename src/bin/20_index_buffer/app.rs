use std::collections::BTreeSet;
use std::convert::TryInto;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Arc;
use std::{mem, process, ptr, slice};

use erupt::utils::surface;
use erupt::vk::{
    make_api_version, AccessFlags, ApplicationInfoBuilder, AttachmentDescriptionBuilder,
    AttachmentLoadOp, AttachmentReferenceBuilder, AttachmentStoreOp, BlendFactor, BlendOp, Buffer,
    BufferCopyBuilder, BufferCreateInfoBuilder, BufferUsageFlags, ClearColorValue, ClearValue,
    ColorComponentFlags, ColorSpaceKHR, CommandBuffer, CommandBufferAllocateInfoBuilder,
    CommandBufferBeginInfoBuilder, CommandBufferLevel, CommandBufferUsageFlags, CommandPool,
    CommandPoolCreateInfoBuilder, ComponentMappingBuilder, ComponentSwizzle,
    CompositeAlphaFlagBitsKHR, CullModeFlags, DeviceCreateInfoBuilder, DeviceMemory,
    DeviceQueueCreateInfoBuilder, DeviceSize, Extent2D, Extent2DBuilder, FenceCreateFlags,
    FenceCreateInfoBuilder, Format, Framebuffer, FramebufferCreateInfoBuilder, FrontFace,
    GraphicsPipelineCreateInfoBuilder, Image, ImageAspectFlags, ImageLayout,
    ImageSubresourceRangeBuilder, ImageUsageFlags, ImageView, ImageViewCreateInfoBuilder,
    ImageViewType, IndexType, InstanceCreateInfoBuilder, LogicOp, MemoryAllocateInfoBuilder,
    MemoryPropertyFlags, Offset2DBuilder, PhysicalDevice, PhysicalDeviceFeaturesBuilder, Pipeline,
    PipelineBindPoint, PipelineColorBlendAttachmentStateBuilder,
    PipelineColorBlendStateCreateInfoBuilder, PipelineInputAssemblyStateCreateInfoBuilder,
    PipelineLayout, PipelineLayoutCreateInfoBuilder, PipelineMultisampleStateCreateInfoBuilder,
    PipelineRasterizationStateCreateInfoBuilder, PipelineShaderStageCreateInfoBuilder,
    PipelineStageFlags, PipelineVertexInputStateCreateInfoBuilder,
    PipelineViewportStateCreateInfoBuilder, PolygonMode, PresentInfoKHRBuilder, PresentModeKHR,
    PrimitiveTopology, Queue, Rect2DBuilder, RenderPass, RenderPassBeginInfoBuilder,
    RenderPassCreateInfoBuilder, SampleCountFlagBits, SemaphoreCreateInfoBuilder, ShaderModule,
    ShaderModuleCreateInfoBuilder, ShaderStageFlagBits, SharingMode, SubmitInfoBuilder,
    SubpassContents, SubpassDependencyBuilder, SubpassDescriptionBuilder, SurfaceCapabilitiesKHR,
    SurfaceFormatKHR, SurfaceKHR, SwapchainCreateInfoKHRBuilder, ViewportBuilder, API_VERSION_1_0,
    KHR_SWAPCHAIN_EXTENSION_NAME, SUBPASS_EXTERNAL,
};
use erupt::{vk, DeviceLoader, EntryLoader, InstanceLoader, SmallVec};
use vulkan_tutorial_erupt::{VkError, VkResult};
use winit::dpi::{LogicalPosition, LogicalSize};
use winit::error::OsError;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use crate::app::buffer::Buffers;
use crate::app::pipeline::RenderPipeline;
use crate::app::queue::{CompleteQueueFamilyIndices, QueueFamilyIndices, Queues};
use crate::app::swapchain::{Swapchain, SwapchainSupportDetails};
use crate::app::sync::SyncObjects;
use crate::app::vertex::Vertex;

mod buffer;
#[cfg(debug_assertions)]
mod debug;
mod pipeline;
mod queue;
mod swapchain;
mod sync;
mod vertex;

const APP_NAME: &str = "Vulkan";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

const DEVICE_EXTENSIONS: &[*const c_char] = &[KHR_SWAPCHAIN_EXTENSION_NAME];

const VERTEX_SHADER_CODE: &[u8] =
    include_bytes!("../../../shaders/17_shader_vertexbuffer.vert.spv");
const FRAGMENT_SHADER_CODE: &[u8] =
    include_bytes!("../../../shaders/17_shader_vertexbuffer.frag.spv");

const MAX_FRAMES_IN_FLIGHT: usize = 2;

// Fields are in reverse order to drop in the correct one.
#[derive(Debug)]
pub struct VulkanApp {
    sync: SyncObjects<MAX_FRAMES_IN_FLIGHT>,
    command_buffers: SmallVec<CommandBuffer>,
    buffers: Buffers,
    command_pool: CommandPool,
    pipeline: RenderPipeline,
    framebuffers: Vec<Framebuffer>,
    swapchain: Swapchain,
    queues: Queues,
    device: Arc<DeviceLoader>,
    physical_device: PhysicalDevice,
    surface: SurfaceKHR,
    #[cfg(debug_assertions)]
    debug_messenger: debug::Messenger,
    instance: Arc<InstanceLoader>,
    _entry: EntryLoader,
}

impl VulkanApp {
    pub fn new(window: &Window) -> VkResult<Self> {
        let entry = EntryLoader::new()?;
        let instance = Self::create_instance(window, &entry)?;

        #[cfg(debug_assertions)]
        let debug_messenger = debug::Messenger::new(&instance)?;

        let surface = Self::create_surface(window, &instance)?;

        let (physical_device, queue_family_indices, swapchain_support) =
            Self::pick_physical_device(&instance, surface)?;
        let (device, queues) =
            Self::create_logical_device(&instance, physical_device, queue_family_indices)?;

        let swapchain = Self::create_swapchain(
            window,
            surface,
            queue_family_indices,
            &swapchain_support,
            &device,
        )?;

        let render_pass = Self::create_render_pass(&device, swapchain.image_format)?;
        let (pipeline_layout, graphics_pipeline) =
            Self::create_graphics_pipeline(&device, swapchain.extent, render_pass)?;
        let pipeline = RenderPipeline {
            pipeline: graphics_pipeline,
            layout: pipeline_layout,
            render_pass,
        };

        let framebuffers = Self::create_framebuffers(
            &device,
            swapchain.extent,
            &swapchain.image_views,
            render_pass,
        )?;

        let command_pool = Self::create_command_pool(queue_family_indices, &device)?;
        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(
            &instance,
            physical_device,
            &device,
            queues.graphics,
            command_pool,
        )?;
        let (index_buffer, index_buffer_memory) = Self::create_index_buffer(
            &instance,
            physical_device,
            &device,
            queues.graphics,
            command_pool,
        )?;
        let buffers = Buffers {
            index_memory: index_buffer_memory,
            index: index_buffer,
            vertex_memory: vertex_buffer_memory,
            vertex: vertex_buffer,
        };
        let command_buffers = Self::create_command_buffers(
            &device,
            swapchain.extent,
            render_pass,
            graphics_pipeline,
            &framebuffers,
            command_pool,
            &buffers,
        )?;

        let sync = Self::create_sync_objects(&device, &swapchain.images)?;

        Ok(Self {
            sync,
            command_buffers,
            buffers,
            command_pool,
            pipeline,
            framebuffers,
            swapchain,
            queues,
            device,
            physical_device,
            surface,
            #[cfg(debug_assertions)]
            debug_messenger,
            instance,
            _entry: entry,
        })
    }

    pub fn build_window(event_loop: &EventLoop<()>) -> Result<Window, OsError> {
        let window = WindowBuilder::new()
            .with_title(APP_NAME)
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .with_resizable(true)
            .build(event_loop)?;

        if let Some(monitor) = window.current_monitor() {
            let monitor_size = monitor.size().to_logical::<f64>(monitor.scale_factor());

            window.set_outer_position(LogicalPosition::new(
                (monitor_size.width - f64::from(WINDOW_WIDTH)) / 2.0,
                (monitor_size.height - f64::from(WINDOW_HEIGHT)) / 2.0,
            ));
        }

        Ok(window)
    }

    pub fn run(mut self, event_loop: EventLoop<()>, window: Window) -> ! {
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            if let Err(err) = self.process_event(&window, event, control_flow) {
                eprintln!("Error: {:?}", color_eyre::Report::new(err));
                process::exit(1);
            }
        })
    }

    fn create_instance(window: &Window, entry: &EntryLoader) -> VkResult<Arc<InstanceLoader>> {
        #[cfg(debug_assertions)]
        use erupt::ExtendableFromConst;

        #[cfg(debug_assertions)]
        if !debug::check_validation_layer_support(entry)? {
            return Err(VkError::ValidationLayerUnavailable);
        }

        let app_name = CString::new(APP_NAME).unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        let app_info = ApplicationInfoBuilder::new()
            .application_name(&app_name)
            .application_version(make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(make_api_version(0, 1, 0, 0))
            .api_version(API_VERSION_1_0);

        let extensions = Self::required_extensions(window)?;

        let create_info = InstanceCreateInfoBuilder::new()
            .application_info(&app_info)
            .enabled_extension_names(&extensions);

        #[cfg(debug_assertions)]
        let debug_create_info = debug::messenger_create_info();
        #[cfg(debug_assertions)]
        let create_info = create_info
            .enabled_layer_names(debug::VALIDATION_LAYERS)
            .extend_from(&debug_create_info);

        Ok(Arc::new(unsafe {
            InstanceLoader::new(entry, &create_info, None)
        }?))
    }

    fn required_extensions(window: &Window) -> VkResult<Vec<*const c_char>> {
        let extensions = surface::enumerate_required_extensions(window).result()?;

        #[cfg(debug_assertions)]
        let mut extensions = extensions;
        #[cfg(debug_assertions)]
        extensions.extend(debug::EXTENSIONS);

        Ok(extensions)
    }

    fn create_surface(window: &Window, instance: &InstanceLoader) -> VkResult<SurfaceKHR> {
        Ok(unsafe { surface::create_surface(instance, window, None) }.result()?)
    }

    fn pick_physical_device(
        instance: &InstanceLoader,
        surface: SurfaceKHR,
    ) -> VkResult<(
        PhysicalDevice,
        CompleteQueueFamilyIndices,
        SwapchainSupportDetails,
    )> {
        let devices = unsafe { instance.enumerate_physical_devices(None) }.result()?;

        if devices.is_empty() {
            return Err(VkError::NoVulkanGpu);
        }

        for device in devices {
            if let Some((indices, swapchain_support)) =
                Self::is_device_suitable(instance, surface, device)?
            {
                return Ok((device, indices, swapchain_support));
            }
        }

        Err(VkError::NoSuitableGpu)
    }

    fn is_device_suitable(
        instance: &InstanceLoader,
        surface: SurfaceKHR,
        device: PhysicalDevice,
    ) -> VkResult<Option<(CompleteQueueFamilyIndices, SwapchainSupportDetails)>> {
        let indices = QueueFamilyIndices::new(instance, surface, device)?;

        if let Some(indices) = indices.complete() {
            if Self::check_device_extension_support(instance, device)? {
                let swapchain_support = SwapchainSupportDetails::new(instance, surface, device)?;

                Ok(swapchain_support
                    .is_adequate()
                    .then(|| (indices, swapchain_support)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    fn check_device_extension_support(
        instance: &InstanceLoader,
        device: PhysicalDevice,
    ) -> VkResult<bool> {
        let available_extensions =
            unsafe { instance.enumerate_device_extension_properties(device, None, None) }
                .result()?;

        let required_extensions = DEVICE_EXTENSIONS
            .iter()
            .map(|ptr| unsafe { CStr::from_ptr(*ptr) });

        Ok(check_support(
            available_extensions
                .iter()
                .map(|extension| unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) }),
            required_extensions,
        ))
    }

    fn create_logical_device(
        instance: &InstanceLoader,
        physical_device: PhysicalDevice,
        indices: CompleteQueueFamilyIndices,
    ) -> VkResult<(Arc<DeviceLoader>, Queues)> {
        let unique_queue_families =
            BTreeSet::from([indices.graphics_family(), indices.present_family()]);

        let queue_priority = 1.0;
        let queue_create_infos = unique_queue_families
            .into_iter()
            .map(|queue_family| {
                DeviceQueueCreateInfoBuilder::new()
                    .queue_family_index(queue_family)
                    .queue_priorities(slice::from_ref(&queue_priority))
            })
            .collect::<Vec<_>>();

        let device_features = PhysicalDeviceFeaturesBuilder::new();

        let create_info = DeviceCreateInfoBuilder::new()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(DEVICE_EXTENSIONS);

        #[cfg(debug_assertions)]
        let create_info = create_info.enabled_layer_names(debug::VALIDATION_LAYERS);

        let device =
            Arc::new(unsafe { DeviceLoader::new(instance, physical_device, &create_info, None) }?);

        let queues = Queues {
            graphics: unsafe { device.get_device_queue(indices.graphics_family(), 0) },
            present: unsafe { device.get_device_queue(indices.present_family(), 0) },
        };

        Ok((device, queues))
    }

    fn create_swapchain(
        window: &Window,
        surface: SurfaceKHR,
        indices: CompleteQueueFamilyIndices,
        swapchain_support: &SwapchainSupportDetails,
        device: &DeviceLoader,
    ) -> VkResult<Swapchain> {
        let capabilities = swapchain_support.capabilities();

        let surface_format = Self::choose_swap_surface_format(swapchain_support.formats());
        let present_mode = Self::choose_swap_present_mode(swapchain_support.present_modes());
        let extent = Self::choose_swap_extent(window, capabilities);

        let image_count = if capabilities.max_image_count > 0 {
            capabilities
                .max_image_count
                .min(capabilities.min_image_count + 1)
        } else {
            capabilities.min_image_count + 1
        };

        let indices = [indices.graphics_family(), indices.present_family()];
        let (sharing_mode, indices) = if indices[0] == indices[1] {
            (SharingMode::EXCLUSIVE, &[][..])
        } else {
            (SharingMode::CONCURRENT, &indices[..])
        };

        let create_info = SwapchainCreateInfoKHRBuilder::new()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(sharing_mode)
            .queue_family_indices(indices)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe { device.create_swapchain_khr(&create_info, None) }.result()?;

        let images = unsafe { device.get_swapchain_images_khr(swapchain, None) }.result()?;
        let image_format = surface_format.format;
        let image_views = Self::create_image_views(device, &images, image_format)?;

        Ok(Swapchain {
            image_views,
            extent,
            image_format,
            images,
            swapchain,
        })
    }

    fn choose_swap_surface_format(available_formats: &[SurfaceFormatKHR]) -> SurfaceFormatKHR {
        *available_formats
            .iter()
            .find(|format| {
                format.format == Format::B8G8R8A8_SRGB
                    && format.color_space == ColorSpaceKHR::SRGB_NONLINEAR_KHR
            })
            .unwrap_or(&available_formats[0])
    }

    fn choose_swap_present_mode(available_present_modes: &[PresentModeKHR]) -> PresentModeKHR {
        *available_present_modes
            .iter()
            .find(|present_mode| **present_mode == PresentModeKHR::MAILBOX_KHR)
            .unwrap_or(&PresentModeKHR::FIFO_KHR)
    }

    fn choose_swap_extent(window: &Window, capabilities: &SurfaceCapabilitiesKHR) -> Extent2D {
        if capabilities.current_extent.width == u32::MAX {
            let size = window.inner_size();

            *Extent2DBuilder::new()
                .width(size.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ))
                .height(size.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ))
        } else {
            capabilities.current_extent
        }
    }

    fn create_image_views(
        device: &DeviceLoader,
        images: &[Image],
        image_format: Format,
    ) -> VkResult<Vec<ImageView>> {
        images
            .iter()
            .map(|image| {
                let create_info = ImageViewCreateInfoBuilder::new()
                    .image(*image)
                    .view_type(ImageViewType::_2D)
                    .format(image_format)
                    .components(
                        *ComponentMappingBuilder::new()
                            .r(ComponentSwizzle::IDENTITY)
                            .g(ComponentSwizzle::IDENTITY)
                            .b(ComponentSwizzle::IDENTITY)
                            .a(ComponentSwizzle::IDENTITY),
                    )
                    .subresource_range(
                        *ImageSubresourceRangeBuilder::new()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );
                unsafe { device.create_image_view(&create_info, None) }.map_err(VkError::Vk)
            })
            .collect()
    }

    fn create_render_pass(device: &DeviceLoader, image_format: Format) -> VkResult<RenderPass> {
        let color_attachment = AttachmentDescriptionBuilder::new()
            .format(image_format)
            .samples(SampleCountFlagBits::_1)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = AttachmentReferenceBuilder::new()
            .attachment(0)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = SubpassDescriptionBuilder::new()
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .color_attachments(slice::from_ref(&color_attachment_ref));

        let dependency = SubpassDependencyBuilder::new()
            .src_subpass(SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(AccessFlags::NONE_KHR)
            .dst_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(AccessFlags::COLOR_ATTACHMENT_WRITE);

        let render_pass_info = RenderPassCreateInfoBuilder::new()
            .attachments(slice::from_ref(&color_attachment))
            .subpasses(slice::from_ref(&subpass))
            .dependencies(slice::from_ref(&dependency));

        Ok(unsafe { device.create_render_pass(&render_pass_info, None) }.result()?)
    }

    fn create_graphics_pipeline(
        device: &DeviceLoader,
        extent: Extent2D,
        render_pass: RenderPass,
    ) -> VkResult<(PipelineLayout, Pipeline)> {
        let vertex_shader_module = Self::create_shader_module(device, VERTEX_SHADER_CODE)?;
        let fragment_shader_module = Self::create_shader_module(device, FRAGMENT_SHADER_CODE)?;

        let name = CString::new("main").unwrap();

        let vertex_shader_stage_info = PipelineShaderStageCreateInfoBuilder::new()
            .stage(ShaderStageFlagBits::VERTEX)
            .module(vertex_shader_module)
            .name(&name);

        let fragment_shader_stage_info = PipelineShaderStageCreateInfoBuilder::new()
            .stage(ShaderStageFlagBits::FRAGMENT)
            .module(fragment_shader_module)
            .name(&name);

        let shader_stages = [vertex_shader_stage_info, fragment_shader_stage_info];

        let binding_description = Vertex::binding_description();
        let attribute_descriptions = Vertex::attribute_descriptions();
        let vertex_input_info = PipelineVertexInputStateCreateInfoBuilder::new()
            .vertex_binding_descriptions(slice::from_ref(&binding_description))
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = PipelineInputAssemblyStateCreateInfoBuilder::new()
            .topology(PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = ViewportBuilder::new()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = Rect2DBuilder::new()
            .offset(*Offset2DBuilder::new().x(0).y(0))
            .extent(extent);

        let viewport_state = PipelineViewportStateCreateInfoBuilder::new()
            .viewports(slice::from_ref(&viewport))
            .scissors(slice::from_ref(&scissor));

        let rasterizer = PipelineRasterizationStateCreateInfoBuilder::new()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(CullModeFlags::BACK)
            .front_face(FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = PipelineMultisampleStateCreateInfoBuilder::new()
            .sample_shading_enable(false)
            .rasterization_samples(SampleCountFlagBits::_1)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        let color_blend_attachment = PipelineColorBlendAttachmentStateBuilder::new()
            .color_write_mask(
                ColorComponentFlags::R
                    | ColorComponentFlags::G
                    | ColorComponentFlags::B
                    | ColorComponentFlags::A,
            )
            .blend_enable(false)
            .src_color_blend_factor(BlendFactor::ONE)
            .dst_color_blend_factor(BlendFactor::ZERO)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::ONE)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .alpha_blend_op(BlendOp::ADD);

        let color_blending = PipelineColorBlendStateCreateInfoBuilder::new()
            .logic_op_enable(false)
            .logic_op(LogicOp::COPY)
            .attachments(slice::from_ref(&color_blend_attachment))
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let pipeline_layout_info = PipelineLayoutCreateInfoBuilder::new();

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }.result()?;

        let pipeline_info = GraphicsPipelineCreateInfoBuilder::new()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)
            .base_pipeline_index(-1);

        let graphics_pipeline = unsafe {
            device.create_graphics_pipelines(None, slice::from_ref(&pipeline_info), None)
        }
        .result()?[0];

        unsafe { device.destroy_shader_module(Some(fragment_shader_module), None) };
        unsafe { device.destroy_shader_module(Some(vertex_shader_module), None) };

        Ok((pipeline_layout, graphics_pipeline))
    }

    fn create_shader_module(device: &DeviceLoader, code: &[u8]) -> VkResult<ShaderModule> {
        let code = unsafe { slice::from_raw_parts::<u32>(code.as_ptr().cast(), code.len() / 4) };
        let create_info = ShaderModuleCreateInfoBuilder::new().code(code);

        Ok(unsafe { device.create_shader_module(&create_info, None) }.result()?)
    }

    fn create_framebuffers(
        device: &DeviceLoader,
        extent: Extent2D,
        image_views: &[ImageView],
        render_pass: RenderPass,
    ) -> VkResult<Vec<Framebuffer>> {
        image_views
            .iter()
            .map(|image_view| {
                let framebuffer_info = FramebufferCreateInfoBuilder::new()
                    .render_pass(render_pass)
                    .attachments(slice::from_ref(image_view))
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                unsafe { device.create_framebuffer(&framebuffer_info, None) }.map_err(VkError::Vk)
            })
            .collect()
    }

    fn create_command_pool(
        indices: CompleteQueueFamilyIndices,
        device: &DeviceLoader,
    ) -> VkResult<CommandPool> {
        let pool_info =
            CommandPoolCreateInfoBuilder::new().queue_family_index(indices.graphics_family());

        Ok(unsafe { device.create_command_pool(&pool_info, None) }.result()?)
    }

    fn create_vertex_buffer(
        instance: &InstanceLoader,
        physical_device: PhysicalDevice,
        device: &DeviceLoader,
        graphics_queue: Queue,
        command_pool: CommandPool,
    ) -> VkResult<(Buffer, DeviceMemory)> {
        let buffer_size = (mem::size_of::<Vertex>() * Vertex::VERTICES.len()) as DeviceSize;

        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let mut data = ptr::null_mut::<Vertex>();
        unsafe {
            device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    None,
                    ptr::addr_of_mut!(data).cast(),
                )
                .result()?;
            data.copy_from_nonoverlapping(Vertex::VERTICES.as_ptr(), Vertex::VERTICES.len());
            device.unmap_memory(staging_buffer_memory);
        }

        let (vertex_buffer, vertex_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER,
            MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        Self::copy_buffer(
            device,
            graphics_queue,
            command_pool,
            staging_buffer,
            vertex_buffer,
            buffer_size,
        )?;

        unsafe { device.destroy_buffer(Some(staging_buffer), None) };
        unsafe { device.free_memory(Some(staging_buffer_memory), None) };

        Ok((vertex_buffer, vertex_buffer_memory))
    }

    fn create_index_buffer(
        instance: &InstanceLoader,
        physical_device: PhysicalDevice,
        device: &DeviceLoader,
        graphics_queue: Queue,
        command_pool: CommandPool,
    ) -> VkResult<(Buffer, DeviceMemory)> {
        let buffer_size = (mem::size_of::<u16>() * Vertex::INDICES.len()) as DeviceSize;

        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let mut data = ptr::null_mut::<u16>();
        unsafe {
            device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    None,
                    ptr::addr_of_mut!(data).cast(),
                )
                .result()?;
            data.copy_from_nonoverlapping(Vertex::INDICES.as_ptr(), Vertex::INDICES.len());
            device.unmap_memory(staging_buffer_memory);
        }

        let (index_buffer, index_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::INDEX_BUFFER,
            MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        Self::copy_buffer(
            device,
            graphics_queue,
            command_pool,
            staging_buffer,
            index_buffer,
            buffer_size,
        )?;

        unsafe { device.destroy_buffer(Some(staging_buffer), None) };
        unsafe { device.free_memory(Some(staging_buffer_memory), None) };

        Ok((index_buffer, index_buffer_memory))
    }

    fn create_buffer(
        instance: &InstanceLoader,
        physical_device: PhysicalDevice,
        device: &DeviceLoader,
        size: DeviceSize,
        usage: BufferUsageFlags,
        properties: MemoryPropertyFlags,
    ) -> VkResult<(Buffer, DeviceMemory)> {
        let buffer_info = BufferCreateInfoBuilder::new()
            .size(size)
            .usage(usage)
            .sharing_mode(SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None) }.result()?;
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let alloc_info = MemoryAllocateInfoBuilder::new()
            .allocation_size(memory_requirements.size)
            .memory_type_index(Self::find_memory_type(
                instance,
                physical_device,
                memory_requirements.memory_type_bits,
                properties,
            )?);

        let buffer_memory = unsafe { device.allocate_memory(&alloc_info, None) }.result()?;
        unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0) }.result()?;

        Ok((buffer, buffer_memory))
    }

    fn find_memory_type(
        instance: &InstanceLoader,
        physical_device: PhysicalDevice,
        type_filter: u32,
        properties: MemoryPropertyFlags,
    ) -> VkResult<u32> {
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        (0..memory_properties.memory_type_count)
            .into_iter()
            .find(|&i| {
                (type_filter & (1 << i)) != 0
                    && memory_properties.memory_types[i as usize]
                        .property_flags
                        .contains(properties)
            })
            .ok_or(VkError::NoSuitableMemoryType)
    }

    fn copy_buffer(
        device: &DeviceLoader,
        graphics_queue: Queue,
        command_pool: CommandPool,
        src: Buffer,
        dst: Buffer,
        size: DeviceSize,
    ) -> VkResult<()> {
        let alloc_info = CommandBufferAllocateInfoBuilder::new()
            .level(CommandBufferLevel::PRIMARY)
            .command_pool(command_pool)
            .command_buffer_count(1);

        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info) }.result()?[0];

        let begin_info =
            CommandBufferBeginInfoBuilder::new().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }.result()?;

        let copy_region = BufferCopyBuilder::new().size(size);
        unsafe { device.cmd_copy_buffer(command_buffer, src, dst, slice::from_ref(&copy_region)) };

        unsafe { device.end_command_buffer(command_buffer) }.result()?;

        let submit_info =
            SubmitInfoBuilder::new().command_buffers(slice::from_ref(&command_buffer));

        unsafe { device.queue_submit(graphics_queue, slice::from_ref(&submit_info), None) }
            .result()?;
        unsafe { device.queue_wait_idle(graphics_queue) }.result()?;

        unsafe { device.free_command_buffers(command_pool, slice::from_ref(&command_buffer)) };

        Ok(())
    }

    fn create_command_buffers(
        device: &DeviceLoader,
        extent: Extent2D,
        render_pass: RenderPass,
        graphics_pipeline: Pipeline,
        framebuffers: &[Framebuffer],
        command_pool: CommandPool,
        buffers: &Buffers,
    ) -> VkResult<SmallVec<CommandBuffer>> {
        let alloc_info = CommandBufferAllocateInfoBuilder::new()
            .command_pool(command_pool)
            .level(CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len().try_into().unwrap());

        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }.result()?;

        for (&command_buffer, &framebuffer) in command_buffers.iter().zip(framebuffers) {
            let begin_info = CommandBufferBeginInfoBuilder::new();
            unsafe { device.begin_command_buffer(command_buffer, &begin_info) }.result()?;

            let clear_color = ClearValue {
                color: ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            };

            let render_pass_info = RenderPassBeginInfoBuilder::new()
                .render_pass(render_pass)
                .framebuffer(framebuffer)
                .render_area(
                    *Rect2DBuilder::new()
                        .offset(*Offset2DBuilder::new().x(0).y(0))
                        .extent(extent),
                )
                .clear_values(slice::from_ref(&clear_color));

            unsafe {
                device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_info,
                    SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(
                    command_buffer,
                    PipelineBindPoint::GRAPHICS,
                    graphics_pipeline,
                );
                device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    slice::from_ref(&buffers.vertex),
                    &[0],
                );
                device.cmd_bind_index_buffer(command_buffer, buffers.index, 0, IndexType::UINT16);

                device.cmd_draw_indexed(
                    command_buffer,
                    Vertex::INDICES.len().try_into().unwrap(),
                    1,
                    0,
                    0,
                    0,
                );
                device.cmd_end_render_pass(command_buffer);

                device.end_command_buffer(command_buffer).result()?;
            }
        }

        Ok(command_buffers)
    }

    fn create_sync_objects(
        device: &DeviceLoader,
        images: &[Image],
    ) -> VkResult<SyncObjects<MAX_FRAMES_IN_FLIGHT>> {
        fn create_objects<T>(
            create_fn: impl Fn() -> VkResult<T>,
        ) -> VkResult<[T; MAX_FRAMES_IN_FLIGHT]>
        where
            T: Default + Copy,
        {
            let mut objects = [T::default(); MAX_FRAMES_IN_FLIGHT];
            for object in &mut objects {
                *object = create_fn()?;
            }
            Ok(objects)
        }

        let semaphore_info = SemaphoreCreateInfoBuilder::new();
        let create_semaphore =
            || unsafe { device.create_semaphore(&semaphore_info, None) }.map_err(VkError::Vk);

        let fence_info = FenceCreateInfoBuilder::new().flags(FenceCreateFlags::SIGNALED);
        let create_fence =
            || unsafe { device.create_fence(&fence_info, None) }.map_err(VkError::Vk);

        Ok(SyncObjects {
            image_available_semaphores: create_objects(create_semaphore)?,
            render_finished_semaphores: create_objects(create_semaphore)?,
            in_flight_fences: create_objects(create_fence)?,
            images_in_flight: vec![None; images.len()],
            current_frame: 0,
        })
    }

    fn process_event(
        &mut self,
        window: &Window,
        event: Event<()>,
        control_flow: &mut ControlFlow,
    ) -> VkResult<()> {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => {
                    if let (Some(VirtualKeyCode::Escape), ElementState::Released) =
                        (input.virtual_keycode, input.state)
                    {
                        *control_flow = ControlFlow::Exit;
                    }
                }
                WindowEvent::Resized(new_inner_size)
                | WindowEvent::ScaleFactorChanged {
                    new_inner_size: &mut new_inner_size,
                    ..
                } => {
                    if self.swapchain.extent.width != new_inner_size.width
                        || self.swapchain.extent.height != new_inner_size.height
                    {
                        self.recreate_swapchain(window)?;
                    }
                }
                _ => (),
            },
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => self.draw_frame(window)?,
            Event::LoopDestroyed => unsafe { self.device.device_wait_idle() }.result()?,
            _ => (),
        }

        Ok(())
    }

    fn draw_frame(&mut self, window: &Window) -> VkResult<()> {
        unsafe {
            self.device.wait_for_fences(
                slice::from_ref(&self.sync.in_flight_fences[self.sync.current_frame]),
                true,
                u64::MAX,
            )
        }
        .result()?;

        let image_index_result = unsafe {
            self.device.acquire_next_image_khr(
                self.swapchain.swapchain,
                u64::MAX,
                Some(self.sync.image_available_semaphores[self.sync.current_frame]),
                None,
            )
        }
        .result();
        let image_index = match image_index_result {
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain(window)?;
                return Ok(());
            }
            result => result,
        }?;

        if let Some(image_in_flight) = self.sync.images_in_flight[image_index as usize]
            .replace(self.sync.in_flight_fences[self.sync.current_frame])
        {
            unsafe {
                self.device
                    .wait_for_fences(slice::from_ref(&image_in_flight), true, u64::MAX)
            }
            .result()?;
        }

        let submit_info = SubmitInfoBuilder::new()
            .wait_semaphores(slice::from_ref(
                &self.sync.image_available_semaphores[self.sync.current_frame],
            ))
            .wait_dst_stage_mask(slice::from_ref(
                &PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ))
            .command_buffers(slice::from_ref(&self.command_buffers[image_index as usize]))
            .signal_semaphores(slice::from_ref(
                &self.sync.render_finished_semaphores[self.sync.current_frame],
            ));

        unsafe {
            self.device.reset_fences(slice::from_ref(
                &self.sync.in_flight_fences[self.sync.current_frame],
            ))
        }
        .result()?;

        unsafe {
            self.device.queue_submit(
                self.queues.graphics,
                slice::from_ref(&submit_info),
                Some(self.sync.in_flight_fences[self.sync.current_frame]),
            )
        }
        .result()?;

        let present_info = PresentInfoKHRBuilder::new()
            .wait_semaphores(slice::from_ref(
                &self.sync.render_finished_semaphores[self.sync.current_frame],
            ))
            .swapchains(slice::from_ref(&self.swapchain.swapchain))
            .image_indices(slice::from_ref(&image_index));

        let present_result = unsafe {
            self.device
                .queue_present_khr(self.queues.present, &present_info)
        };

        if present_result.raw == vk::Result::ERROR_OUT_OF_DATE_KHR
            || present_result.raw == vk::Result::SUBOPTIMAL_KHR
        {
            self.recreate_swapchain(window)?;
        } else {
            present_result.result()?;
        }

        self.sync.current_frame = (self.sync.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    fn recreate_swapchain(&mut self, window: &Window) -> VkResult<()> {
        unsafe { self.device.device_wait_idle() }.result()?;

        unsafe { self.cleanup_swapchain() };

        let queue_family_indices =
            QueueFamilyIndices::new(&self.instance, self.surface, self.physical_device)?
                .complete()
                .ok_or(VkError::NoSuitableGpu)?;
        let swapchain_support =
            SwapchainSupportDetails::new(&self.instance, self.surface, self.physical_device)?;

        self.swapchain = Self::create_swapchain(
            window,
            self.surface,
            queue_family_indices,
            &swapchain_support,
            &self.device,
        )?;

        self.pipeline.render_pass =
            Self::create_render_pass(&self.device, self.swapchain.image_format)?;
        let (pipeline_layout, graphics_pipeline) = Self::create_graphics_pipeline(
            &self.device,
            self.swapchain.extent,
            self.pipeline.render_pass,
        )?;
        self.pipeline.layout = pipeline_layout;
        self.pipeline.pipeline = graphics_pipeline;

        self.framebuffers = Self::create_framebuffers(
            &self.device,
            self.swapchain.extent,
            &self.swapchain.image_views,
            self.pipeline.render_pass,
        )?;

        self.command_buffers = Self::create_command_buffers(
            &self.device,
            self.swapchain.extent,
            self.pipeline.render_pass,
            self.pipeline.pipeline,
            &self.framebuffers,
            self.command_pool,
            &self.buffers,
        )?;

        self.sync
            .images_in_flight
            .resize(self.swapchain.images.len(), None);

        Ok(())
    }

    unsafe fn cleanup_swapchain(&mut self) {
        for framebuffer in &self.framebuffers {
            self.device.destroy_framebuffer(Some(*framebuffer), None);
        }

        self.device
            .free_command_buffers(self.command_pool, &self.command_buffers);

        self.device
            .destroy_pipeline(Some(self.pipeline.pipeline), None);

        self.device
            .destroy_pipeline_layout(Some(self.pipeline.layout), None);

        self.device
            .destroy_render_pass(Some(self.pipeline.render_pass), None);

        for image_view in &self.swapchain.image_views {
            self.device.destroy_image_view(Some(*image_view), None);
        }

        self.device
            .destroy_swapchain_khr(Some(self.swapchain.swapchain), None);
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_fence(Some(self.sync.in_flight_fences[i]), None);
                self.device
                    .destroy_semaphore(Some(self.sync.render_finished_semaphores[i]), None);
                self.device
                    .destroy_semaphore(Some(self.sync.image_available_semaphores[i]), None);
            }

            self.cleanup_swapchain();

            self.device.destroy_buffer(Some(self.buffers.index), None);

            self.device
                .free_memory(Some(self.buffers.index_memory), None);

            self.device.destroy_buffer(Some(self.buffers.vertex), None);

            self.device
                .free_memory(Some(self.buffers.vertex_memory), None);

            self.device
                .destroy_command_pool(Some(self.command_pool), None);

            self.device.destroy_device(None);

            self.instance.destroy_surface_khr(Some(self.surface), None);

            #[cfg(debug_assertions)]
            self.debug_messenger.destroy(&self.instance);

            self.instance.destroy_instance(None);
        }
    }
}

fn check_support<'a>(
    available: impl IntoIterator<Item = &'a CStr>,
    required: impl IntoIterator<Item = &'a CStr>,
) -> bool {
    let mut required = required.into_iter().collect::<BTreeSet<_>>();

    for available in available {
        required.remove(available);
    }

    required.is_empty()
}
