use glam::Mat4;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct UniformBufferObject {
    model: Mat4,
    view: Mat4,
    projection: Mat4,
}

impl UniformBufferObject {
    pub const fn new(model: Mat4, view: Mat4, projection: Mat4) -> Self {
        Self {
            model,
            view,
            projection,
        }
    }
}
