use std::convert::TryFrom;

use erupt::vk::{PhysicalDevice, Queue, QueueFlags, SurfaceKHR};
use erupt::InstanceLoader;
use vulkan_tutorial_erupt::VkResult;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Queues {
    pub present: Queue,
    pub graphics: Queue,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn new(
        instance: &InstanceLoader,
        surface: SurfaceKHR,
        device: PhysicalDevice,
    ) -> VkResult<Self> {
        let mut indices = Self {
            graphics_family: None,
            present_family: None,
        };

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(device, None) };

        for (i, queue_family) in queue_families.into_iter().enumerate() {
            let i = u32::try_from(i).unwrap();

            if queue_family.queue_flags.contains(QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(i);
            }

            if unsafe { instance.get_physical_device_surface_support_khr(device, i, surface) }
                .result()?
            {
                indices.present_family = Some(i);
            }

            if indices.is_complete() {
                break;
            }
        }

        Ok(indices)
    }

    pub fn complete(self) -> Option<CompleteQueueFamilyIndices> {
        Some(CompleteQueueFamilyIndices {
            graphics_family: self.graphics_family?,
            present_family: self.present_family?,
        })
    }

    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CompleteQueueFamilyIndices {
    graphics_family: u32,
    present_family: u32,
}

impl CompleteQueueFamilyIndices {
    pub fn graphics_family(self) -> u32 {
        self.graphics_family
    }

    pub fn present_family(self) -> u32 {
        self.present_family
    }
}
