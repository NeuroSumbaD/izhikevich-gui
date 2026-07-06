/*
# Izhikevich-neuron model implementations

    Provides multiple implementations for simulating the Izhikevich neuron model
    with floating-point and fixed-point arithmetic.
*/

use std::collections::VecDeque;

// use crate::qmath::FixedPoint;

#[derive(Clone, Copy, PartialEq)]
pub struct NeuronParams {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub dt: f32,
    pub input_current: f32,
    pub duration: f32,
}

impl Default for NeuronParams {
    fn default() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
            dt: 0.25,
            input_current: 10.0,
            duration: 250.0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct NeuralState {
    pub t: f32,
    pub v: f32,
    pub u: f32,
}

pub struct LiveSimulation {
    pub window_samples: usize,
    pub state: NeuralState,
    pub history: VecDeque<NeuralState>,
    pub spike_times: VecDeque<f32>,
}


impl LiveSimulation {
    pub fn new(params: &NeuronParams) -> Self {
        let v = -65.0_f32;
        let u = params.b * v;
        let window_samples = ((params.duration / params.dt.max(0.001)).ceil() as usize).max(2) + 1;

        let mut simulation = Self {
            window_samples,
            state: NeuralState { t: 0.0, v, u },
            history: VecDeque::with_capacity(window_samples),
            spike_times: VecDeque::new(),
        };

        simulation.history.push_back(NeuralState { t: 0.0, v, u });

        while simulation.state.t < params.duration {
            simulation.step(params);
        }

        simulation
    }

    pub fn step(&mut self, params: &NeuronParams) {
        let dt = params.dt.max(0.001);
        let (next_v, next_u) = rk4_step(self.state.v, self.state.u, params, dt);

        self.state.t += dt;
        self.state.v = next_v;
        self.state.u = next_u;

        if self.state.v >= 30.0 {
            self.history.push_back(NeuralState {
                t: self.state.t,
                v: 30.0,
                u: self.state.u,
            });
            self.spike_times.push_back(self.state.t);
            self.state.v = params.c;
            self.state.u += params.d;
        }

        self.history.push_back(NeuralState {
            t: self.state.t,
            v: self.state.v,
            u: self.state.u,
        });

        while self.history.len() > self.window_samples {
            self.history.pop_front();
        }

        let window_start = self.history.front().map(|sample| sample.t).unwrap_or(self.state.t);
        while self.spike_times.front().is_some_and(|spike_time| *spike_time < window_start) {
            self.spike_times.pop_front();
        }
    }
}

fn rk4_step(v: f32, u: f32, params: &NeuronParams, dt: f32) -> (f32, f32) {
    let k1 = derivatives(v, u, params);
    let k2 = derivatives(
        v + 0.5 * dt * k1.0,
        u + 0.5 * dt * k1.1,
        params,
    );
    let k3 = derivatives(
        v + 0.5 * dt * k2.0,
        u + 0.5 * dt * k2.1,
        params,
    );
    let k4 = derivatives(v + dt * k3.0, u + dt * k3.1, params);

    let next_v = v + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
    let next_u = u + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;

    (next_v, next_u)
}

fn derivatives(v: f32, u: f32, params: &NeuronParams) -> (f32, f32) {
    let dv = 0.04 * v * v + 5.0 * v + 140.0 - u + params.input_current;
    let du = params.a * (params.b * v - u);

    (dv, du)
}
