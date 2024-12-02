#![warn(clippy::all, clippy::pedantic)]

mod texture;

use std::sync::Arc;

use pollster::FutureExt;
use wgpu::{util::DeviceExt, Adapter, Device, Instance, Queue, Surface, SurfaceCapabilities};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::{Window, WindowId},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
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
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

const VERTICES_PENTAGON: &[Vertex] = &[
    // Changed
    Vertex {
        position: [-0.086_824_1, 0.492_403_86, 0.0],
        tex_coords: [0.413_175_9, 0.007_596_14],
    }, // A
    Vertex {
        position: [-0.495_134_06, 0.069_586_47, 0.0],
        tex_coords: [0.004_865_944_4, 0.430_413_54],
    }, // B
    Vertex {
        position: [-0.219_185_49, -0.449_397_06, 0.0],
        tex_coords: [0.280_814_53, 0.949_397],
    }, // C
    Vertex {
        position: [0.359_669_98, -0.347_329_1, 0.0],
        tex_coords: [0.85967, 0.847_329_14],
    }, // D
    Vertex {
        position: [0.441_473_72, 0.234_735_9, 0.0],
        tex_coords: [0.941_473_7, 0.265_264_1],
    }, // E
];

const VERTICES_STAR: &[Vertex] = &[
    Vertex {
        position: [-0.004_444_44, 0.3, 0.0],
        tex_coords: [0.5, 0.225],
    },
    Vertex {
        position: [-0.08, 0.137_777_78, 0.0],
        tex_coords: [0.42, 0.4075],
    },
    Vertex {
        position: [-0.255_555_56, 0.111_111_11, 0.0],
        tex_coords: [0.244_444_44, 0.4375],
    },
    Vertex {
        position: [-0.126_666_67, -0.013_333_33, 0.0],
        tex_coords: [0.373_333_33, 0.5775],
    },
    Vertex {
        position: [-0.153_333_33, -0.186_666_68, 0.0],
        tex_coords: [0.346_666_67, 0.7725],
    },
    Vertex {
        position: [0.002_222_22, -0.104_444_44, 0.0],
        tex_coords: [0.502_222_22, 0.68],
    },
    Vertex {
        position: [0.117_777_78, -0.182_222_22, 0.0],
        tex_coords: [0.617_777_78, 0.7675],
    },
    Vertex {
        position: [0.128_888_89, -0.008_888_89, 0.0],
        tex_coords: [0.628_888_9, 0.5725],
    },
    Vertex {
        position: [0.253_333_33, 0.117_777_79, 0.0],
        tex_coords: [0.753_333_33, 0.43],
    },
    Vertex {
        position: [0.077_777_78, 0.14, 0.0],
        tex_coords: [0.577_777_8, 0.405],
    },
];

const INDICES_PENTAGON: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];
const INDICES_STAR: &[u16] = &[
    0, 1, 9, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 1, 7, 9, 1, 5, 7, 1, 3, 5,
];

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

// Needed for Rust to store our data as expected by the shaders
#[repr(C)]
// Needed so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // We cannot use cgmath with bytemuch directly, so con the Matrix4 to a 4x4 f32 array
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

#[expect(clippy::struct_excessive_bools)]
struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    KeyCode::KeyW | KeyCode::ArrowUp => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyA | KeyCode::ArrowLeft => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyS | KeyCode::ArrowDown => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyD | KeyCode::ArrowRight => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when the camera gets too close to the centre of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the forward/backward is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and the eye so that it doesn't change.  The
            // eye, therefore, still lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            // Rescale the distance between the target and the eye so that it doesn't change.  The
            // eye, therefore, still lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}

struct State<'a> {
    surface: wgpu::Surface<'a>,
    surface_configured: bool,
    device: wgpu::Device,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_controller: CameraController,
    camera_bind_group: wgpu::BindGroup,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
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
    window: Arc<Window>,
    num_indices_0: u32,
    num_indices_1: u32,
    diffuse_bind_group_0: wgpu::BindGroup,
    diffuse_bind_group_1: wgpu::BindGroup,
    modifiers: ModifiersState,
}

impl State<'_> {
    #[allow(clippy::too_many_lines)]
    fn new(window: Window) -> Self {
        let window_arc = Arc::new(window);
        let size = window_arc.inner_size();
        let instance = Self::create_gpu_instance();

        let surface = instance.create_surface(window_arc.clone()).unwrap();

        let adapter = Self::create_adapter(&instance, &surface);

        let (device, queue) = Self::create_device(&adapter);

        let surface_caps = surface.get_capabilities(&adapter);
        let config = Self::create_surface_config(size, &surface_caps);
        surface.configure(&device, &config);

        let diffuse_bytes_0 = include_bytes!("happy-tree.bdff8a19.png");
        let diffuse_texture_0 = texture::Texture::from_bytes(
            &device,
            &queue,
            diffuse_bytes_0,
            "happy-tree.bdff8a19.png",
        )
        .unwrap();
        let texture_bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture_0.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture_0.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });
        let diffuse_bytes_1 = include_bytes!("catfornarak.png");
        let diffuse_texture_1 =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes_1, "catfornarak.png")
                .unwrap();
        let texture_bind_group_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture_1.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture_1.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let aspect: f32;
        #[allow(clippy::cast_precision_loss)]
        {
            aspect = config.width as f32 / config.height as f32;
        }

        let camera = Camera {
            // position the camera 1 unit up and 2 units back
            // +z is out of the screen
            eye: (0.0, 1.0, 2.0).into(),
            // camera looks at the origin
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let camera_controller = CameraController::new(0.2);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });
        let shader_0 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RegularShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let shader_1 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ChallengeShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader_challenge.wgsl").into()),
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout_0, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline_0 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Regular Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_0,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_0,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
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
            cache: None,
        });
        let render_pipeline_1 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Challenge Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_1,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_1,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
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
            cache: None,
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

        let clear_colour = wgpu::Color {
            r: 0.1,
            g: 0.2,
            b: 0.3,
            a: 1.0,
        };

        let num_indices_0 = u32::try_from(INDICES_PENTAGON.len())
            .expect("Pentagon should have few enough indices to be represented as a u32");
        let num_indices_1 = u32::try_from(INDICES_STAR.len())
            .expect("Star should have few enough indices to be represented as a u32");

        Self {
            window: window_arc,
            surface,
            surface_configured: false,
            device,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            queue,
            config,
            clear_colour,
            size,
            use_initial_render_pipeline: true,
            render_pipeline_0,
            render_pipeline_1,
            vertex_buffer_0,
            vertex_buffer_1,
            index_buffer_0,
            index_buffer_1,
            num_indices_0,
            num_indices_1,
            diffuse_bind_group_0,
            diffuse_bind_group_1,
            modifiers: ModifiersState::empty(),
        }
    }

    fn create_surface_config(
        size: PhysicalSize<u32>,
        capabilities: &SurfaceCapabilities,
    ) -> wgpu::SurfaceConfiguration {
        let surface_format = capabilities
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(capabilities.formats[0]);

        debug_assert!(size.width != 0);
        debug_assert!(size.height != 0);

        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: capabilities.present_modes[0],
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        }
    }

    fn create_device(adapter: &Adapter) -> (Device, Queue) {
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .block_on()
            .unwrap()
    }

    fn create_adapter(instance: &Instance, surface: &Surface) -> Adapter {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(surface),
                force_fallback_adapter: false,
            })
            .block_on()
            .unwrap()
    }

    fn create_gpu_instance() -> Instance {
        Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        })
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
        let mods = self.modifiers;
        if mods == ModifiersState::SUPER {
            return false;
        }
        self.camera_controller.process_events(event)
    }

    #[allow(clippy::unused_self)]
    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

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
                render_pass.set_bind_group(0, &self.diffuse_bind_group_0, &[]);
                render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.vertex_buffer_0.slice(..));
                render_pass
                    .set_index_buffer(self.index_buffer_0.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.num_indices_0, 0, 0..1);
            } else {
                render_pass.set_pipeline(&self.render_pipeline_1);
                render_pass.set_bind_group(0, &self.diffuse_bind_group_1, &[]);
                render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
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

struct StateApplication<'a> {
    state: Option<State<'a>>,
}

impl StateApplication<'_> {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for StateApplication<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_inner_size(winit::dpi::LogicalSize::new(450, 400))
                    .with_title("Learn wgpu"),
            )
            .unwrap();
        self.state = Some(State::new(window));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let window = self.state.as_ref().unwrap().window();

        if window.id() == window_id && !self.state.as_mut().unwrap().input(&event) {
            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    self.state.as_mut().unwrap().surface_configured = true;
                    self.state.as_mut().unwrap().resize(physical_size);
                }
                WindowEvent::RedrawRequested => {
                    self.state.as_mut().unwrap().window().request_redraw();

                    if !self.state.as_ref().unwrap().surface_configured {
                        return;
                    }

                    self.state.as_mut().unwrap().update();
                    match self.state.as_mut().unwrap().render() {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let current_size = self.state.as_ref().unwrap().size;
                            self.state.as_mut().unwrap().resize(current_size);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("OutOfMemory");
                            event_loop.exit();
                        }
                        Err(wgpu::SurfaceError::Timeout) => {
                            log::error!("Surface timeout");
                            event_loop.exit();
                        }
                    }
                }
                WindowEvent::ModifiersChanged(modifiers) => {
                    self.state.as_mut().unwrap().modifiers = modifiers.state();
                    log::trace!(
                        "Modifiers changed to {:?}",
                        self.state.as_ref().unwrap().modifiers
                    );
                }
                WindowEvent::KeyboardInput {
                    event,
                    is_synthetic: false,
                    ..
                } => {
                    let mods = self.state.as_ref().unwrap().modifiers;
                    if mods == ModifiersState::SUPER {
                        if let PhysicalKey::Code(KeyCode::KeyQ | KeyCode::KeyW) = event.physical_key
                        {
                            event_loop.exit();
                        }
                    } else {
                        match event.physical_key {
                            PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                            PhysicalKey::Code(KeyCode::Space) => {
                                self.state.as_mut().unwrap().use_initial_render_pipeline =
                                    !self.state.as_ref().unwrap().use_initial_render_pipeline;
                            }
                            _ => {}
                        }
                    }
                }

                _ => {}
            }
        }
    }
}

///
/// # Panics
/// Panics if unable to log to the console
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch="wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn‘t initialise logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new().expect("Unexpected issue creating event loop");
    let mut window_state = StateApplication::new();

    let _ = event_loop.run_app(&mut window_state);

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
}
