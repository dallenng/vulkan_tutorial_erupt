# Vulkan Tutorial erupt

This is an implementation of the [Vulkan Tutorial](https://vulkan-tutorial.com) in Rust using the bindings provided
by [erupt](https://lib.rs/crates/erupt).

## Status

Only missing multisampling chapter, all other chapters are complete.

## Requirements

- [cargo](https://www.rust-lang.org/learn/get-started) with a recent stable toolchain
- A graphics card and driver compatible with Vulkan
- Vulkan SDK (validation layers not needed if compiling with --release)
- glslc

For more information on Vulkan SDK and glslc, see [here](https://vulkan-tutorial.com/Development_environment).

## Usage

Downloads the assets and compile the shaders with the shell scripts :

```shell
./download_assets.sh
./compile.sh
```

The scripts assume your current working directory is the root directory of the project.

Run any chapter with :

```shell
cargo run --bin <chapter_name>
```

Set `RUST_LOG=debug` to see all validation output.

All implemented chapters :

chapter_name | code | reference
-------------|------|----------
00_base_code | [00_base_code](src/bin/00_base_code) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Base_code)
01_instance_creation | [01_instance_creation](src/bin/01_instance_creation) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Instance)
02_validation_layers | [02_validation_layers](src/bin/02_validation_layers) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Validation_layers)
03_physical_device_selection | [03_physical_device_selection](src/bin/03_physical_device_selection) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Physical_devices_and_queue_families)
04_logical_device | [04_logical_device](src/bin/04_logical_device) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Logical_device_and_queues)
05_window_surface | [05_window_surface](src/bin/05_window_surface) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Window_surface)
06_swap_chain_creation | [06_swap_chain_creation](src/bin/06_swap_chain_creation) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain)
07_image_views | [07_image_views](src/bin/07_image_views) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Image_views)
08_graphics_pipeline | [08_graphics_pipeline](src/bin/08_graphics_pipeline) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Introduction)
09_shader_modules | [09_shader_modules](src/bin/09_shader_modules) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Shader_modules)
10_fixed_functions | [10_fixed_functions](src/bin/10_fixed_functions) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Fixed_functions)
11_render_passes | [11_render_passes](src/bin/11_render_passes) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Render_passes)
12_graphics_pipeline_complete | [12_graphics_pipeline_complete](src/bin/12_graphics_pipeline_complete) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Conclusion)
13_framebuffers | [13_framebuffers](src/bin/13_framebuffers) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Framebuffers)
14_command_buffers | [14_command_buffers](src/bin/14_command_buffers) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers)
15_hello_triangle | [15_hello_triangle](src/bin/15_hello_triangle) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Rendering_and_presentation)
16_swap_chain_recreation | [16_swap_chain_recreation](src/bin/16_swap_chain_recreation) | [Link](https://vulkan-tutorial.com/Drawing_a_triangle/Swap_chain_recreation)
17_vertex_input | [17_vertex_input](src/bin/17_vertex_input) | [Link](https://vulkan-tutorial.com/Vertex_buffers/Vertex_input_description)
18_vertex_buffer | [18_vertex_buffer](src/bin/18_vertex_buffer) | [Link](https://vulkan-tutorial.com/Vertex_buffers/Vertex_buffer_creation)
19_staging_buffer | [19_staging_buffer](src/bin/19_staging_buffer) | [Link](https://vulkan-tutorial.com/Vertex_buffers/Staging_buffer)
20_index_buffer | [20_index_buffer](src/bin/20_index_buffer) | [Link](https://vulkan-tutorial.com/Vertex_buffers/Index_buffer)
21_descriptor_layout | [21_descriptor_layout](src/bin/21_descriptor_layout) | [Link](https://vulkan-tutorial.com/Uniform_buffers/Descriptor_layout_and_buffer)
22_descriptor_sets | [22_descriptor_sets](src/bin/22_descriptor_sets) | [Link](https://vulkan-tutorial.com/Uniform_buffers/Descriptor_pool_and_sets)
23_texture_image | [23_texture_image](src/bin/23_texture_image) | [Link](https://vulkan-tutorial.com/Texture_mapping/Images)
24_sampler | [24_sampler](src/bin/24_sampler) | [Link](https://vulkan-tutorial.com/Texture_mapping/Image_view_and_sampler)
25_texture_mapping | [25_texture_mapping](src/bin/25_texture_mapping) | [Link](https://vulkan-tutorial.com/Texture_mapping/Combined_image_sampler)
26_depth_buffering | [26_depth_buffering](src/bin/26_depth_buffering) | [Link](https://vulkan-tutorial.com/Depth_buffering)
27_model_loading | [27_model_loading](src/bin/27_model_loading) | [Link](https://vulkan-tutorial.com/Loading_models)
28_mipmapping | [28_mipmapping](src/bin/28_mipmapping) | [Link](https://vulkan-tutorial.com/Generating_Mipmaps)

## Other Rust implementations of the tutorial

- [unknownue/vulkan-tutorial-rust](https://github.com/unknownue/vulkan-tutorial-rust) using ash
- [adrien-ben/vulkan-tutorial-rs](https://github.com/adrien-ben/vulkan-tutorial-rs) using ash
- [bwasty/vulkan-tutorial-rs](https://github.com/bwasty/vulkan-tutorial-rs) using vulkano

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this repo by you, as
defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
