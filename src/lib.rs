// The wasm-pack uses wasm-bindgen to build and generate JavaScript binding file.
// Import the wasm-bindgen crate.
use wasm_bindgen::prelude::*;
use ggblas::batched_sgemm;
use ggblas::f16::{batched_sgemm_t_f16_pure, batched_sgemm_t_f16_mixed};

extern crate web_sys;
use web_sys::console;

use half::f16;


pub struct Timer<'a> {
    name: &'a str,
}

impl<'a> Timer<'a> {
    pub fn new(name: &'a str) -> Timer<'a> {
        console::time_with_label(name);
        Timer { name }
    }
}

impl<'a> Drop for Timer<'a> {
    fn drop(&mut self) {
        console::time_end_with_label(self.name);
    }
}

const M: usize = 6;
const N: usize = 768 * 3;
const K: usize = 768;
// Our Add function
// wasm-pack requires "exported" functions
// to include #[wasm_bindgen]
#[wasm_bindgen]
pub fn run(){
    init_panic_hook();
    let a = vec![0.0f32; M * K];
    let b = vec![0.0f32; N * K];
    let mut c = vec![0.0f32; M * N];

    let warmups = 5;
    let runs = 10;

    // Simple (2, 2) x (2, 2)
    for _ in 0..warmups{
        batched_sgemm(&a, &b, &mut c, M, N, K);
    }
    let name = format!("matmul-{M}-{N}-{K}");
    for _ in 0..runs{
        let _timer = Timer::new(&name);
        batched_sgemm(&a, &b, &mut c, M, N, K);
    }
}

#[wasm_bindgen]
pub fn run_f16(){
    init_panic_hook();
    let a = vec![f16::from_f32(0.0f32); M * K];
    let b = vec![f16::from_f32(0.0f32); N * K];
    let mut c = vec![f16::from_f32(0.0f32); M * N];

    let warmups = 5;
    let runs = 10;

    // Simple (2, 2) x (2, 2)
    for _ in 0..warmups{
        batched_sgemm_t_f16_pure(&a, &b, &mut c, M, N, K);
    }
    let name = format!("matmul-f16-pure-{M}-{N}-{K}");
    for _ in 0..runs{
        let _timer = Timer::new(&name);
        batched_sgemm_t_f16_pure(&a, &b, &mut c, M, N, K);
    }
}

#[wasm_bindgen]
pub fn run_f16_mixed(){
    init_panic_hook();
    let a = vec![0.0f32; M * K];
    let b = vec![f16::from_f32(0.0f32); N * K];
    let mut c = vec![0.0f32; M * N];

    let warmups = 5;
    let runs = 10;

    // Simple (2, 2) x (2, 2)
    for _ in 0..warmups{
        batched_sgemm_t_f16_mixed(&a, &b, &mut c, M, N, K);
    }
    let name = format!("matmul-f16-mixed-{M}-{N}-{K}");
    for _ in 0..runs{
        let _timer = Timer::new(&name);
        batched_sgemm_t_f16_mixed(&a, &b, &mut c, M, N, K);
    }
}



pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

fn matmul(lhs: &[f32], rhs: &[f32], dst: &mut [f32], m: usize, n: usize, k: usize){
    use gemm::{gemm, Parallelism};
        // let (b, m, n, k) = self.0;
        // // let lhs = &lhs[lhs_l.start_offset()..];
        // // let rhs = &rhs[rhs_l.start_offset()..];

        // let lhs_stride = lhs_l.stride();
        // let rhs_stride = rhs_l.stride();
        // let rank = lhs_stride.len();
        // let lhs_cs = lhs_stride[rank - 1];
        // let lhs_rs = lhs_stride[rank - 2];

        // let rhs_cs = rhs_stride[rank - 1];
        // let rhs_rs = rhs_stride[rank - 2];

        // let a_skip: usize = match lhs_stride[..rank - 2] {
        //     [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
        //     [stride] => stride,
        //     [] => m * k,
        //     _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?,
        // };
        // let b_skip: usize = match rhs_stride[..rank - 2] {
        //     [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
        //     [stride] => stride,
        //     [] => n * k,
        //     _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?,
        // };
        // let c_skip: usize = m * n;

        // let dst_shape: Shape = (m, n).into();
        // let dst_strides = dst_shape.stride_contiguous();
        // let dst_rs = dst_strides[0];
        // let dst_cs = dst_strides[1];

        // let mut dst = vec![T::zero(); b * m * n];
        // let num_threads = crate::utils::get_num_threads();
        // let parallelism = if num_threads > 1 {
        //     Parallelism::Rayon(num_threads)
        // } else {
        // let parallelism = Parallelism::None;
        let parallelism = Parallelism::Rayon(8);
        let parallelism = Parallelism::Rayon(128);
        for step in 0..1 {
            let a_skip = 0;
            let b_skip = 0;
            let c_skip = 0;

            let lhs_rs = K;
            let lhs_cs = 1;
            let rhs_rs = N;
            let rhs_cs = 1;
            let dst_rs = N;
            let dst_cs = 1;
            let lhs_p = &lhs[step * a_skip..];
            let rhs_p = &rhs[step * b_skip..];
            let dst_p = &mut dst[step * c_skip..];
            unsafe {
                gemm(
                    /* m: usize = */ m,
                    /* n: usize = */ n,
                    /* k: usize = */ k,
                    /* dst: *mut T = */ dst_p.as_mut_ptr(),
                    /* dst_cs: isize = */ dst_cs as isize,
                    /* dst_rs: isize = */ dst_rs as isize,
                    /* read_dst: bool = */ false,
                    /* lhs: *const T = */ lhs_p.as_ptr(),
                    /* lhs_cs: isize = */ lhs_cs as isize,
                    /* lhs_rs: isize = */ lhs_rs as isize,
                    /* rhs: *const T = */ rhs_p.as_ptr(),
                    /* rhs_cs: isize = */ rhs_cs as isize,
                    /* rhs_rs: isize = */ rhs_rs as isize,
                    /* alpha: T = */ 0.0,
                    /* beta: T = */ 1.0,
                    /* conj_dst: bool = */ false,
                    /* conj_lhs: bool = */ false,
                    /* conj_rhs: bool = */ false,
                    parallelism,
                )
            }
        }
}

#[wasm_bindgen]
pub fn run_faer(){
    init_panic_hook();
    let a = vec![0.0f32; M * K];
    let b = vec![0.0f32; N * K];
    let mut c = vec![0.0f32; M * N];

    let warmups = 5;
    let runs = 10;
    for _ in 0..warmups{
        matmul(&a, &b, &mut c, M, N, K);
    }
    let name = format!("matmul-gemm-{M}-{N}-{K}");
    for _ in 0..runs{
        let _timer = Timer::new(&name);
        matmul(&a, &b, &mut c, M, N, K);
    }
}


#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}
