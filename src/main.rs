#![warn(clippy::all, clippy::pedantic)]

use learn_wgpu::run;

fn main() {
    // do not use block_on inside an async function, when supportgin WASM
    pollster::block_on(run());
}
