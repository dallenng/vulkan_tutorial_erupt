use erupt::vk::{Pipeline, PipelineLayout, RenderPass};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct RenderPipeline {
    pub pipeline: Pipeline,
    pub layout: PipelineLayout,
    pub render_pass: RenderPass,
}
