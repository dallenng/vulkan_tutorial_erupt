use winit::event_loop::EventLoop;

use crate::app::VulkanApp;

mod app;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let _logger = flexi_logger::Logger::try_with_env_or_str("info")?.start()?;

    let event_loop = EventLoop::new();
    let window = VulkanApp::build_window(&event_loop)?;

    let app = VulkanApp::new(&window)?;
    app.run(event_loop, window);
}
