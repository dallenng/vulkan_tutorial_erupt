use erupt::vk::{
    Extent2D, Format, Image, PhysicalDevice, PresentModeKHR, SurfaceCapabilitiesKHR,
    SurfaceFormatKHR, SurfaceKHR, SwapchainKHR,
};
use erupt::{InstanceLoader, SmallVec};
use vulkan_tutorial_erupt::VkResult;

#[derive(Debug, Clone)]
pub struct Swapchain {
    pub extent: Extent2D,
    pub image_format: Format,
    pub images: SmallVec<Image>,
    pub swapchain: SwapchainKHR,
}

#[derive(Debug, Clone)]
pub struct SwapchainSupportDetails {
    capabilities: SurfaceCapabilitiesKHR,
    formats: SmallVec<SurfaceFormatKHR>,
    present_modes: SmallVec<PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(
        instance: &InstanceLoader,
        surface: SurfaceKHR,
        device: PhysicalDevice,
    ) -> VkResult<Self> {
        let capabilities =
            unsafe { instance.get_physical_device_surface_capabilities_khr(device, surface) }
                .result()?;

        let formats =
            unsafe { instance.get_physical_device_surface_formats_khr(device, surface, None) }
                .result()?;

        let present_modes = unsafe {
            instance.get_physical_device_surface_present_modes_khr(device, surface, None)
        }
        .result()?;

        Ok(Self {
            capabilities,
            formats,
            present_modes,
        })
    }

    pub fn is_adequate(&self) -> bool {
        !self.formats.is_empty() && !self.present_modes.is_empty()
    }

    pub fn capabilities(&self) -> &SurfaceCapabilitiesKHR {
        &self.capabilities
    }

    pub fn formats(&self) -> &[SurfaceFormatKHR] {
        &self.formats
    }

    pub fn present_modes(&self) -> &[PresentModeKHR] {
        &self.present_modes
    }
}
