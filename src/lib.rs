use std::error::Error;
use std::fmt;

use erupt::utils::loading::EntryLoaderError;
use erupt::{vk, LoaderError};
use image::ImageError;
use tobj::LoadError;

pub type VkResult<T> = Result<T, VkError>;

#[derive(Debug)]
pub enum VkError {
    EntryLoader(EntryLoaderError),
    Loader(LoaderError),
    Vk(vk::Result),
    Image(ImageError),
    ObjLoad(LoadError),
    ValidationLayerUnavailable,
    NoVulkanGpu,
    NoSuitableGpu,
    NoSuitableMemoryType,
    NoSupportedFormat,
    UnsupportedLayoutTransition,
    UnsupportedLinearBlitting,
}

impl From<EntryLoaderError> for VkError {
    fn from(err: EntryLoaderError) -> Self {
        Self::EntryLoader(err)
    }
}

impl From<LoaderError> for VkError {
    fn from(err: LoaderError) -> Self {
        Self::Loader(err)
    }
}

impl From<vk::Result> for VkError {
    fn from(err: vk::Result) -> Self {
        Self::Vk(err)
    }
}

impl From<ImageError> for VkError {
    fn from(err: ImageError) -> Self {
        Self::Image(err)
    }
}

impl From<LoadError> for VkError {
    fn from(err: LoadError) -> Self {
        Self::ObjLoad(err)
    }
}

impl fmt::Display for VkError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VkError::EntryLoader(_) => f.write_str("entry loader error"),
            VkError::Loader(_) => f.write_str("loader error"),
            VkError::Vk(err) => write!(f, "vulkan error {}", err.0),
            VkError::Image(_) => f.write_str("image error"),
            VkError::ObjLoad(_) => f.write_str("obj load error"),
            VkError::ValidationLayerUnavailable => {
                f.write_str("validation layers requested, but not available")
            }
            VkError::NoVulkanGpu => f.write_str("failed to find GPUs with Vulkan support"),
            VkError::NoSuitableGpu => f.write_str("failed to find a suitable GPU"),
            VkError::NoSuitableMemoryType => f.write_str("failed to find suitable memory type"),
            VkError::NoSupportedFormat => f.write_str("failed to find supported format"),
            VkError::UnsupportedLayoutTransition => f.write_str("unsupported layout transition"),
            VkError::UnsupportedLinearBlitting => {
                f.write_str("texture image format does not support linear blitting!")
            }
        }
    }
}

impl Error for VkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            VkError::EntryLoader(err) => Some(err),
            VkError::Loader(err) => Some(err),
            VkError::Vk(err) => Some(err),
            VkError::Image(err) => Some(err),
            VkError::ObjLoad(err) => Some(err),
            VkError::ValidationLayerUnavailable
            | VkError::NoVulkanGpu
            | VkError::NoSuitableGpu
            | VkError::NoSuitableMemoryType
            | VkError::NoSupportedFormat
            | VkError::UnsupportedLayoutTransition
            | VkError::UnsupportedLinearBlitting => None,
        }
    }
}
