#![warn(clippy::all, clippy::pedantic)]

use std::{iter, mem};

use wgpu::{MemoryHints, PipelineCompilationOptions};

extern crate framework;

async fn create_adapter(instance: &wgpu::Instance) -> wgpu::Adapter {
    instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap()
}

async fn create_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                // WebGL does not support all of wgpu's features, so if we are building for the web,
                // we will need to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: MemoryHints::default(),
            },
            None, // Trace path
        )
        .await
        .unwrap()
}

fn render(
    colour: &[f64; 3],
    render_target: &framework::Texture,
    render_pipeline: &wgpu::RenderPipeline,
    encoder: &mut wgpu::CommandEncoder,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("GIF Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &render_target.view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: colour[0],
                    g: colour[1],
                    b: colour[2],
                    a: 1.0,
                }),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    rpass.set_pipeline(render_pipeline);
    rpass.draw(0..3, 0..1);
}

async fn run() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = create_adapter(&instance).await;
    let (device, queue) = create_device(&adapter).await;

    let colours = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.2],
        [0.0, 0.2, 0.2],
        [0.2, 0.2, 0.2],
        [0.0, 0.2, 0.2],
        [0.0, 0.0, 0.2],
        [0.0, 0.0, 0.0],
    ];

    // create a texture for rendering
    let texture_size = 256u32;
    let rt_desc = wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: texture_size,
            height: texture_size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: None,
        view_formats: &[],
    };
    let render_target = framework::Texture::from_descriptor(&device, rt_desc);

    // wgpu requires texture -> buffer copies to be aligned usingwgpu::COPY_BYTES_PER_ROW_ALIGNMENT.
    // Because of this, we will need to save both the `padded_bytes_per_row`, as well ad the
    // `unpadded_bytes_per_row`
    let pixel_size = u32::try_from(mem::size_of::<[u8; 4]>()).expect("Should fit");
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let unpadded_bytes_per_row = pixel_size * texture_size;
    let padding = (align - unpadded_bytes_per_row % align) % align;
    let padded_bytes_per_row = unpadded_bytes_per_row + padding;

    // create a buffer to copy the texture to, so we canget the data
    let buffer_size = wgpu::BufferAddress::from(padded_bytes_per_row * texture_size);
    let buffer_desc = wgpu::BufferDescriptor {
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: Some("Output Buffer"),
        mapped_at_creation: false,
    };
    let output_buffer = device.create_buffer(&buffer_desc);

    // a simple render pipeline that draws a triangle
    let render_pipeline = create_render_pipeline(&device, &render_target);

    let mut frames = Vec::new();

    for c in &colours {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        render(c, &render_target, &render_pipeline, &mut encoder);

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &render_target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(texture_size),
                },
            },
            render_target.desc.size,
        );

        queue.submit(iter::once(encoder.finish()));

        // Create the map request
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        // wait for the GPU to finish
        device.poll(wgpu::Maintain::Wait);

        match rx.receive().await {
            Some(Ok(())) => {
                let padded_data = buffer_slice.get_mapped_range();
                let data = padded_data
                    .chunks(padded_bytes_per_row as _)
                    .flat_map(|chunk| &chunk[..unpadded_bytes_per_row as _])
                    .copied()
                    .collect::<Vec<_>>();
                drop(padded_data);
                output_buffer.unmap();
                frames.push(data);
            }
            _ => eprintln!("Something went wrong"),
        }
    }

    save_gif(
        "output.gif",
        &mut frames,
        10,
        u16::try_from(texture_size).expect("should fit"),
    )
    .unwrap();
}

fn save_gif(path: &str, frames: &mut Vec<Vec<u8>>, speed: i32, size: u16) -> anyhow::Result<()> {
    use gif::{Encoder, Frame, Repeat};

    let mut image = std::fs::File::create(path)?;
    let mut encoder = Encoder::new(&mut image, size, size, &[])?;
    encoder.set_repeat(Repeat::Infinite)?;

    for frame in frames {
        encoder.write_frame(&Frame::from_rgba_speed(size, size, frame, speed))?;
    }

    Ok(())
}

fn create_render_pipeline(
    device: &wgpu::Device,
    target: &framework::Texture,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: Some(&render_pipeline_layout),
        label: Some("Render Pipeline"),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target.desc.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back), //Setting
            //this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
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
    })
}

fn main() {
    pollster::block_on(run());
}
