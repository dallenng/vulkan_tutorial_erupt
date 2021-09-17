use erupt::vk::{Buffer, DeviceMemory};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct Buffers {
    pub index_memory: DeviceMemory,
    pub index: Buffer,
    pub vertex_memory: DeviceMemory,
    pub vertex: Buffer,
}
