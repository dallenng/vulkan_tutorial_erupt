use std::collections::BTreeSet;
use std::ffi::CString;
use std::os::raw::c_char;
use std::slice;
use std::sync::Arc;

use erupt::utils::surface;
use erupt::vk::{
    make_api_version, ApplicationInfoBuilder, DeviceCreateInfoBuilder,
    DeviceQueueCreateInfoBuilder, InstanceCreateInfoBuilder, PhysicalDevice,
    PhysicalDeviceFeaturesBuilder, SurfaceKHR, API_VERSION_1_0,
};
use erupt::{DeviceLoader, EntryLoader, InstanceLoader};
use vulkan_tutorial_erupt::{VkError, VkResult};
use winit::dpi::{LogicalPosition, LogicalSize};
use winit::error::OsError;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use crate::app::queue::{CompleteQueueFamilyIndices, QueueFamilyIndices, Queues};

#[cfg(debug_assertions)]
mod debug;
mod queue;

const APP_NAME: &str = "Vulkan";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

// Fields are in reverse order to drop in the correct one.
#[derive(Debug)]
pub struct VulkanApp {
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

        let (physical_device, queue_family_indices) =
            Self::pick_physical_device(&instance, surface)?;
        let (device, queues) =
            Self::create_logical_device(&instance, physical_device, queue_family_indices)?;

        Ok(Self {
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
    ) -> VkResult<(PhysicalDevice, CompleteQueueFamilyIndices)> {
        let devices = unsafe { instance.enumerate_physical_devices(None) }.result()?;

        if devices.is_empty() {
            return Err(VkError::NoVulkanGpu);
        }

        for device in devices {
            if let Some(indices) = Self::is_device_suitable(instance, surface, device)? {
                return Ok((device, indices));
            }
        }

        Err(VkError::NoSuitableGpu)
    }

    fn is_device_suitable(
        instance: &InstanceLoader,
        surface: SurfaceKHR,
        device: PhysicalDevice,
    ) -> VkResult<Option<CompleteQueueFamilyIndices>> {
        let indices = QueueFamilyIndices::new(instance, surface, device)?;

        Ok(indices.complete())
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
            .enabled_features(&device_features);

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

    fn draw_frame(&mut self) {}
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);

            self.instance.destroy_surface_khr(Some(self.surface), None);

            #[cfg(debug_assertions)]
            self.debug_messenger.destroy(&self.instance);

            self.instance.destroy_instance(None);
        }
    }
}
