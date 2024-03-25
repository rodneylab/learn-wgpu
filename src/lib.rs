use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalPosition,
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [0.0, 0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
];

const VERTICES_PENTAGON: &[Vertex] = &[
    Vertex {
        position: [-0.0868241, 0.49240386, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [-0.49513406, 0.06958647, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [-0.21918549, -0.44939706, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [0.35966998, -0.3473291, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [0.44147372, 0.2347359, 0.0],
        color: [0.5, 0.0, 0.5],
    },
];

const VERTICES_STAR: &[Vertex] = &[
    Vertex {
        position: [-0.00444444, 0.3, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [-0.08, 0.13777778, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [-0.25555556, 0.11111111, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [-0.12666667, -0.01333333, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [-0.15333333, -0.18666667, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [0.00222222, -0.10444444, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [0.11777778, -0.18222222, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [0.12888889, -0.00888889, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [0.25333333, 0.11777778, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [0.07777778, 0.14, 0.0],
        color: [0.5, 0.0, 0.5],
    },
];

const INDICES_PENTAGON: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];
const INDICES_STAR: &[u16] = &[
    0, 1, 9, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 1, 7, 9, 1, 5, 7, 1, 3, 5,
];

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    // render_pipeline: &'a wgpu::RenderPipeline,
    render_pipeline_0: wgpu::RenderPipeline,
    render_pipeline_1: wgpu::RenderPipeline,
    vertex_buffer_0: wgpu::Buffer,
    vertex_buffer_1: wgpu::Buffer,
    index_buffer_0: wgpu::Buffer,
    index_buffer_1: wgpu::Buffer,
    use_initial_render_pipeline: bool,
    clear_colour: wgpu::Color,
    // The window must be declared after the surface so it gets dropped after the surface, as it
    // contains unsafe references to the window's resources.
    window: Window,
    num_indices_0: u32,
    num_indices_1: u32,
}

impl State {
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window, so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            // note enumerate_adapters - used to iterate over all adapters is not available on
            // WASM, request_adapter (used here) is.
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        let shader_0 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let shader_1 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader_challenge.wgsl").into()),
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout 0"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });
        let render_pipeline_0 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline 0"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_0,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_0,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZTION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        let render_pipeline_1 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline 1"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_1,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_1,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZTION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let vertex_buffer_0 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer Pentagon"),
            contents: bytemuck::cast_slice(VERTICES_PENTAGON),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let vertex_buffer_1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer Star"),
            contents: bytemuck::cast_slice(VERTICES_STAR),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer_0 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer Pentagon"),
            contents: bytemuck::cast_slice(INDICES_PENTAGON),
            usage: wgpu::BufferUsages::INDEX,
        });
        let index_buffer_1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer Star"),
            contents: bytemuck::cast_slice(INDICES_STAR),
            usage: wgpu::BufferUsages::INDEX,
        });
        surface.configure(&device, &config);

        let clear_colour = wgpu::Color {
            r: 0.1,
            g: 0.2,
            b: 0.3,
            a: 1.0,
        };

        let num_indices_0 = INDICES_PENTAGON.len() as u32;
        let num_indices_1 = INDICES_STAR.len() as u32;

        Self {
            window,
            surface,
            device,
            queue,
            config,
            clear_colour,
            size,
            //render_pipeline: &render_pipeline_0,
            use_initial_render_pipeline: true,
            render_pipeline_0,
            render_pipeline_1,
            vertex_buffer_0,
            vertex_buffer_1,
            index_buffer_0,
            index_buffer_1,
            num_indices_0,
            num_indices_1,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved {
                position: PhysicalPosition::<f64> { x, y },
                ..
            } => {
                self.clear_colour = wgpu::Color {
                    r: x / self.size.width as f64,
                    g: y / self.size.width as f64,
                    b: 1.0,
                    a: 1.0,
                };

                true
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Space),
                        ..
                    },
                ..
            } => {
                self.use_initial_render_pipeline = !self.use_initial_render_pipeline;
                true
            }

            _ => false,
        }
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_colour),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            if self.use_initial_render_pipeline {
                render_pass.set_pipeline(&self.render_pipeline_0);
                render_pass.set_vertex_buffer(0, self.vertex_buffer_0.slice(..));
                render_pass
                    .set_index_buffer(self.index_buffer_0.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.num_indices_0, 0, 0..1);
            } else {
                render_pass.set_pipeline(&self.render_pipeline_1);
                render_pass.set_vertex_buffer(0, self.vertex_buffer_1.slice(..));
                render_pass
                    .set_index_buffer(self.index_buffer_1.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.num_indices_1, 0, 0..1);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch="wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn‘t initialise logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new().expect("Unexpected issue creating event loop");
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    use winit::dpi::PhysicalSize;
    let _ = window.request_inner_size(PhysicalSize::new(450, 400));

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn‘t append canvas to document body.");
    }

    let mut state = State::new(window).await;

    let _=event_loop.run(move |event,  elwt| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window().id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                                ..
                            },
                        ..
                    } => (*elwt).exit(),
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged {
                         ..
                    } => { /* should now receive a Resized event after scale factor change */
                    }
        //WindowEvent::RedrawRequested(window_id) if window_id == state.window().id() => {
        WindowEvent::RedrawRequested=>{
            state.update();
            match state.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                Err(wgpu::SurfaceError::OutOfMemory) => (*elwt).exit(),
                // All other errors (Outdated, Timeout) should be resolved by the next
                // frame
                Err(e) => eprintln!("{:?}", e),
            }
        }
                    _ => {}
                }
            }
        }
        Event::AboutToWait => {
            // RedrawRequested will only trigger once, unless we manually request it
            state.window().request_redraw();
        }
        _ => {}
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
