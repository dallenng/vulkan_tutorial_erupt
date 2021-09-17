use std::collections::BTreeSet;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::slice;
use std::sync::Arc;

use erupt::utils::surface;
use erupt::vk::{
    make_api_version, ApplicationInfoBuilder, ColorSpaceKHR, ComponentMappingBuilder,
    ComponentSwizzle, CompositeAlphaFlagBitsKHR, DeviceCreateInfoBuilder,
    DeviceQueueCreateInfoBuilder, Extent2D, Extent2DBuilder, Format, Image, ImageAspectFlags,
    ImageSubresourceRangeBuilder, ImageUsageFlags, ImageView, ImageViewCreateInfoBuilder,
    ImageViewType, InstanceCreateInfoBuilder, PhysicalDevice, PhysicalDeviceFeaturesBuilder,
    PresentModeKHR, SharingMode, SurfaceCapabilitiesKHR, SurfaceFormatKHR, SurfaceKHR,
    SwapchainCreateInfoKHRBuilder, API_VERSION_1_0, KHR_SWAPCHAIN_EXTENSION_NAME,
};
use erupt::{DeviceLoader, EntryLoader, InstanceLoader};
use vulkan_tutorial_erupt::{VkError, VkResult};
use winit::dpi::{LogicalPosition, LogicalSize};
use winit::error::OsError;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use crate::app::queue::{CompleteQueueFamilyIndices, QueueFamilyIndices, Queues};
use crate::app::swapchain::{Swapchain, SwapchainSupportDetails};

#[cfg(debug_assertions)]
mod debug;
mod queue;
mod swapchain;

const APP_NAME: &str = "Vulkan";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

const DEVICE_EXTENSIONS: &[*const c_char] = &[KHR_SWAPCHAIN_EXTENSION_NAME];

// Fields are in reverse order to drop in the correct one.
#[derive(Debug)]
pub struct VulkanApp {
    swapchain: Swapchain,
    _queues: Queues,
    device: Arc<DeviceLoader>,
    _physical_device: PhysicalDevice,
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

        Self::create_graphics_pipeline();

        Ok(Self {
            swapchain,
            _queues: queues,
            device,
            _physical_device: physical_device,
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
            .with_resizable(false)
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
                    _ => (),
                },
                Event::MainEventsCleared => window.request_redraw(),
                Event::RedrawRequested(_) => {
                    // Required to drop VulkanApp
                    self.draw_frame();
                }
                _ => (),
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

    fn create_graphics_pipeline() {}

    fn draw_frame(&mut self) {}
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            for image_view in &self.swapchain.image_views {
                self.device.destroy_image_view(Some(*image_view), None);
            }

            self.device
                .destroy_swapchain_khr(Some(self.swapchain.swapchain), None);

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
