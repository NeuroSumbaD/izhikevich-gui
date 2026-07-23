/*
# Izhikevich-neuron model implementations

    Provides multiple implementations for simulating the Izhikevich neuron model
    with floating-point and fixed-point arithmetic.
*/

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Normal};

use crate::qmath::FixedPoint;

#[derive(PartialEq, Serialize, Deserialize, Clone, Copy)]
pub enum NeuralModel {
    FloatingPoint,
    FixedPoint { bit_width: usize, q_width: usize },
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct NeuronParams {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub dt: f32,
    pub rev_e: f32,
    pub rev_l: f32,
    pub rev_i: f32,
    pub ge_bar: f32,
    pub gi_bar: f32,
    pub gl_bar: f32,
    pub model: NeuralModel,
    pub num_neurons: usize,
    pub exc_inputs: Vec<f32>,
    pub duration: f32,
    pub noise_std_devs: Vec<f32>,
    pub rng_seed: [u8; 32],
}

impl Default for NeuronParams {
    fn default() -> Self {
        let mut default_input: Vec<f32> = Vec::with_capacity(10);
        default_input.push(0.1);

        let mut default_noise: Vec<f32> = Vec::with_capacity(10);
        default_noise.push(0.0);

        Self {
            rev_e: 45.0,
            rev_l: -75.0,
            rev_i: -85.0,
            ge_bar: 1.5,
            gi_bar: 1.0,
            gl_bar: 1.0,
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
            dt: 0.25,
            model: NeuralModel::FloatingPoint,
            num_neurons: 1,
            exc_inputs: default_input,
            duration: 250.0,
            noise_std_devs: default_noise,
            rng_seed: [0; 32],
        }
    }
}

struct QParams {
    pub a: FixedPoint,
    pub b: FixedPoint,
    pub dt: FixedPoint,
    pub rev_e: FixedPoint,
    pub rev_l: FixedPoint,
    pub rev_i: FixedPoint,
    pub ge_bar: FixedPoint,
    pub gi_bar: FixedPoint,
    pub gl_bar: FixedPoint,
}

impl QParams {
    fn from_fp(params: &NeuronParams, bit_width: usize, q_width: usize) -> Self {
        Self {
            a: FixedPoint::new(bit_width, q_width, params.a as f64),
            b: FixedPoint::new(bit_width, q_width, params.b as f64),
            dt: FixedPoint::new(bit_width, q_width, params.dt as f64),
            rev_e: FixedPoint::new(bit_width, q_width, params.rev_e as f64),
            rev_l: FixedPoint::new(bit_width, q_width, params.rev_l as f64),
            rev_i: FixedPoint::new(bit_width, q_width, params.rev_i as f64),
            ge_bar: FixedPoint::new(bit_width, q_width, params.ge_bar as f64),
            gi_bar: FixedPoint::new(bit_width, q_width, params.gi_bar as f64),
            gl_bar: FixedPoint::new(bit_width, q_width, params.gl_bar as f64),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct NeuralState {
    pub t: f32,
    pub v: f32,
    pub u: f32,
}


/// # LiveSimulation 
/// Holds and manages the state of a simulation
/// 
/// Note that the parameters are external so they can be smoothly adjusted in
/// real-time external to the simulation.
#[derive(Serialize, Deserialize, Clone)]
pub struct LiveSimulation {
    pub window_samples: usize,
    pub simulation_time: f32,
    pub histories: Vec<VecDeque<NeuralState>>,
    pub spike_times: Vec<VecDeque<f32>>,
}


impl LiveSimulation {
    pub fn new(params: &NeuronParams, rng: &mut SmallRng) -> Self {
        let v = -65.0_f32;
        let u = params.b * v;
        let window_samples = ((params.duration / params.dt.max(0.001)).ceil() as usize).max(2) + 1;

        let mut simulation = Self {
            window_samples,
            simulation_time: 0.0,
            histories: Vec::from_iter(std::iter::repeat_with(|| VecDeque::with_capacity(window_samples)).take(params.num_neurons)),
            spike_times: Vec::from_iter(std::iter::repeat_with(|| VecDeque::new()).take(params.num_neurons)),
        };

        for history in simulation.histories.iter_mut() {
            history.push_back(NeuralState { t: 0.0, v, u });
        }

        while simulation.simulation_time < params.duration {
            simulation.step(params, rng);
        }

        simulation
    }

    pub fn step(&mut self, params: &NeuronParams, rng: &mut SmallRng) {
        let dt = params.dt.max(0.001);
        self.simulation_time += dt;

        for (neuron, history) in self.histories.iter_mut().enumerate() {
            // Step each neuron
            let mut state = history.back().unwrap().clone();
            let mut exc_input = params.exc_inputs.get(neuron).copied().unwrap_or(0.0);
            let noise_std_dev = params.noise_std_devs.get(neuron).copied().unwrap_or(0.0);
            let spike_times = &mut self.spike_times[neuron];

            // Add input noise to the current input
            if noise_std_dev > 0.0 {
                let noise: f32 = Normal::new(0.0, noise_std_dev).unwrap().sample(rng);
                exc_input += noise;
            }

            let (next_v, next_u) = 
                match params.model {
                    NeuralModel::FloatingPoint => rk4_step(state.v, state.u, exc_input, params, dt),
                    NeuralModel::FixedPoint { bit_width, q_width } => {
                        q_step(state.v, state.u, exc_input, params,  bit_width, q_width)
                    }
                };

            state.t += dt;
            state.v = next_v;
            state.u = next_u;

            if state.v >= 30.0 {
                history.push_back(NeuralState {
                    t: state.t,
                    v: 30.0,
                    u: state.u,
                });
                spike_times.push_back(state.t);
                state.v = params.c;
                state.u += params.d;
            }

            history.push_back(NeuralState {
                t: state.t,
                v: state.v,
                u: state.u,
            });

            while history.len() > self.window_samples {
                history.pop_front();
            }

            let window_start = history.front().map(|sample| sample.t).unwrap_or(state.t);
            while spike_times.front().is_some_and(|spike_time| *spike_time < window_start) {
                spike_times.pop_front();
            }

        }
    }
}

/// Runge-Kutta 4th order method for solving the Izhikevich neuron model equations
/// # Returns
///  - next_v: f32 -- next membrane potential
///  - next_u: f32 -- next recovery variable
fn rk4_step(v: f32, u: f32, exc_input: f32, params: &NeuronParams, dt: f32) -> (f32, f32) {
    let k1 = derivatives(v, u, exc_input, params);
    let k2 = derivatives(
        v + 0.5 * dt * k1.0,
        u + 0.5 * dt * k1.1,
        exc_input,
        params,
    );
    let k3 = derivatives(
        v + 0.5 * dt * k2.0,
        u + 0.5 * dt * k2.1,
        exc_input,
        params,
    );
    let k4 = derivatives(v + dt * k3.0, u + dt * k3.1, exc_input, params);

    let next_v = v + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
    let next_u = u + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;

    (next_v, next_u)
}

fn derivatives(v: f32, u: f32, exc_input: f32, params: &NeuronParams) -> (f32, f32) {
    let i_exc = params.ge_bar * (params.rev_e - v) * exc_input;
    let i_leak = params.gl_bar * (params.rev_l - v);
    // TODO: implement FFFB lateral inhibition
    // let i_inh = params.gi_bar * (params.rev_i - v) * exc_input;
    let current = i_exc + i_leak;

    let dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current;
    let du = params.a * (params.b * v - u);

    (dv, du)
}

/// Euler step method implemented with fixed-point arithmetic for the Izhikevich neuron model equations
/// # Arguments
/// * `v` - The current membrane potential
/// * `u` - The current recovery variable
/// * `current` - The current input
/// * `params` - The neuron parameters
/// * `bit_width` - The width of the fixed-point representation
/// * `q_width` - The quantization width of the fixed-point representation
/// # Returns
///  - next_v: f32 -- next membrane potential
///  - next_u: f32 -- next recovery variable
fn q_step(v: f32, u: f32, exc_input: f32, params: &NeuronParams, bit_width: usize, q_width: usize) -> (f32, f32) {
    // convert state variables and parameters to fixed-point representation
    let v_q = FixedPoint::new(bit_width, q_width, v);
    let u_q = FixedPoint::new(bit_width, q_width, u);
    let exc_input_q = FixedPoint::new(bit_width, q_width, exc_input);
    let params_q = QParams::from_fp(params, bit_width, q_width);

    let i_exc_q = (
        params_q.ge_bar * 
        (params_q.rev_e - v_q).truncate(bit_width, q_width) *
        exc_input_q
        ).truncate(bit_width, q_width);
    let i_leak_q = (
        params_q.gl_bar *
        (params_q.rev_l - v_q).truncate(bit_width, q_width)
        ).truncate(bit_width, q_width);
    // TODO: implement FFFB lateral inhibition
    // let i_inh_q = FixedPoint::new(bit_width, q_width, params.gi_bar * (params.rev_i - v) * exc_input);

    let current_q = (i_exc_q + i_leak_q).truncate(bit_width, q_width);
    // let current_q = i_exc_q + i_leak_q + i_inh_q;


    let x_const = FixedPoint::new(bit_width, q_width, 0.04);
    let y_const = FixedPoint::new(bit_width, q_width, 5.0);
    let w_const = FixedPoint::new(bit_width, q_width, 140.0);

    let v_s1 = (v_q * x_const).truncate(bit_width, q_width);
    let v_s2 = (v_s1 + y_const).truncate(bit_width, q_width);
    let v_s3 = (v_s2 * v_q).truncate(bit_width, q_width);
    let v_s4 = (current_q + w_const + v_s3 - u_q).truncate(bit_width, q_width);
    let v_next = ((v_s4 * params_q.dt).truncate(bit_width, q_width) + v_q).truncate(bit_width, q_width);

    let u_s1 = (v_q * params_q.b).truncate(bit_width, q_width);
    let u_s2 = (u_s1 - u_q).truncate(bit_width, q_width);
    let u_s3 = (params_q.a * u_s2).truncate(bit_width, q_width);
    let u_next = ((u_s3 * params_q.dt).truncate(bit_width, q_width) + u_q).truncate(bit_width, q_width);

    (v_next.to_f32(), u_next.to_f32())
}
