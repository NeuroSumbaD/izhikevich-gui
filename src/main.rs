use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use std::collections::VecDeque;
use std::time::Duration;

fn main() -> eframe::Result<()> {
    let mut options = eframe::NativeOptions::default();
    options.viewport.title = Some("Izhikevich Neuron Visualizer".to_string());
    options.viewport.maximized = Some(true);
    eframe::run_native(
        "Izhikevich Neuron Visualizer",
        options,
        Box::new(|_cc| Ok(Box::new(NeuronApp::default()))),
    )
}

#[derive(Clone, Copy, PartialEq)]
struct NeuronParams {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    dt: f32,
    input_current: f32,
    duration: f32,
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

struct Sample {
    t: f32,
    v: f32,
    u: f32,
}

struct LiveSimulation {
    window_samples: usize,
    t: f32,
    v: f32,
    u: f32,
    samples: VecDeque<Sample>,
    euler_samples: VecDeque<Sample>,
    spike_times: VecDeque<f32>,
}

struct NeuronApp {
    params: NeuronParams,
    simulation: LiveSimulation,
    running: bool,
    show_euler: bool,
    target_fps: f32,
    steps_per_frame: u32,
}

impl Default for NeuronApp {
    fn default() -> Self {
        let params = NeuronParams::default();
        let simulation = LiveSimulation::new(&params);

        Self {
            params,
            simulation,
            running: true,
            show_euler: false,
            target_fps: 30.0,
            steps_per_frame: 4,
        }
    }
}

impl eframe::App for NeuronApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let mut needs_reset = false;

        egui::Panel::left("controls")
            .resizable(false)
            .default_size(280.0)
            .show_inside(ui, |ui| {
                ui.heading("Izhikevich Controls");
                ui.label("Adjust the model parameters and watch the RK4 simulation evolve live.");

                ui.separator();

                needs_reset |= slider(ui, "a", &mut self.params.a, 0.001, 0.2);
                needs_reset |= slider(ui, "b", &mut self.params.b, 0.01, 0.5);
                needs_reset |= slider(ui, "c", &mut self.params.c, -80.0, -40.0);
                needs_reset |= slider(ui, "d", &mut self.params.d, 0.0, 20.0);
                needs_reset |= slider(ui, "dt (ms)", &mut self.params.dt, 0.01, 2.0);
                needs_reset |= slider(ui, "Input current", &mut self.params.input_current, 0.0, 50.0);
                needs_reset |= slider(ui, "Window (ms)", &mut self.params.duration, 20.0, 1000.0);
                needs_reset |= slider(ui, "Update FPS", &mut self.target_fps, 1.0, 120.0);
                needs_reset |= slider(ui, "Steps / frame", &mut self.steps_per_frame, 1, 25);

                ui.separator();

                // Show Euler step for comparison
                let atoms = "Show Euler step";
                ui.checkbox(&mut self.show_euler, atoms);

                ui.horizontal(|ui| {
                    if ui
                        .button(if self.running { "Pause" } else { "Resume" })
                        .clicked()
                    {
                        self.running = !self.running;
                    }

                    if ui.button("Reset").clicked() {
                        needs_reset = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Status:");
                    ui.label(if self.running {
                        "running"
                    } else {
                        "paused"
                    });
                });

                ui.label(format!("Live time: {:.2} ms", self.simulation.t));
                ui.label(format!("Samples: {}", self.simulation.samples.len()));
                ui.label(format!("Spikes: {}", self.simulation.spike_times.len()));

                if let Some(last_spike) = self.simulation.spike_times.back() {
                    ui.label(format!("Last spike: {:.2} ms", last_spike));
                } else {
                    ui.label("Last spike: none");
                }

                if let Some(last) = self.simulation.samples.back() {
                    ui.label(format!("Final V: {:.2} mV", last.v));
                    ui.label(format!("Final u: {:.2}", last.u));
                }
            });

        if needs_reset {
            self.simulation = LiveSimulation::new(&self.params);
            self.running = true;
        }

        if self.running {
            for _ in 0..self.steps_per_frame {
                self.simulation.step(&self.params);
            }
            ui.ctx()
                .request_repaint_after(Duration::from_secs_f32(1.0 / self.target_fps.max(1.0)));
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Neuron Response");
            ui.add_space(8.0);

            let voltage_points = PlotPoints::from_iter(
                self.simulation.samples.iter().map(|sample| [sample.t as f64, sample.v as f64]),
            );
            let recovery_points = PlotPoints::from_iter(
                self.simulation.samples.iter().map(|sample| [sample.t as f64, sample.u as f64]),
            );

            Plot::new("Membrane Voltage (mV)")
                .legend(Legend::default())
                .height(260.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("RK4", voltage_points));

                    // If Euler step is enabled
                    if self.show_euler {
                        let euler_points = PlotPoints::from_iter(
                            self.simulation.euler_samples.iter().map(|sample| {
                                [sample.t as f64, sample.v as f64]
                            })
                        );
                        plot_ui.line(Line::new("Euler", euler_points).style(egui_plot::LineStyle::dashed_dense()));
                    }
                });

            ui.add_space(12.0);

            Plot::new("Recovery variable u")
                .legend(Legend::default())
                .height(220.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("RK4", recovery_points));

                    // If Euler step is enabled
                    if self.show_euler {
                        let euler_points = PlotPoints::from_iter(
                            self.simulation.euler_samples.iter().map(|sample| {
                                [sample.t as f64, sample.u as f64]
                            })
                        );
                        plot_ui.line(Line::new("Euler", euler_points).style(egui_plot::LineStyle::dashed_dense()));
                    }
                });
        });
    }
}

impl LiveSimulation {
    fn new(params: &NeuronParams) -> Self {
        let v = -65.0_f32;
        let u = params.b * v;
        let window_samples = ((params.duration / params.dt.max(0.001)).ceil() as usize).max(2) + 1;

        let mut simulation = Self {
            window_samples,
            t: 0.0,
            v,
            u,
            samples: VecDeque::with_capacity(window_samples),
            euler_samples: VecDeque::with_capacity(window_samples),
            spike_times: VecDeque::new(),
        };

        simulation.samples.push_back(Sample { t: 0.0, v, u });
        simulation.euler_samples.push_back(Sample { t: 0.0, v, u });

        while simulation.t < params.duration {
            simulation.step(params);
        }

        simulation
    }

    fn step(&mut self, params: &NeuronParams) {
        let dt = params.dt.max(0.001);
        let (next_v, next_u) = rk4_step(self.v, self.u, params, dt);
        let euler_state = euler_step(self.v, self.u, params, dt);

        self.t += dt;
        self.v = next_v;
        self.u = next_u;
        self.euler_samples.push_back(Sample { t: self.t, v: euler_state.0.min(30.0), u: euler_state.1 });

        if self.v >= 30.0 {
            self.samples.push_back(Sample {
                t: self.t,
                v: 30.0,
                u: self.u,
            });
            self.spike_times.push_back(self.t);
            self.v = params.c;
            self.u += params.d;
        }

        self.samples.push_back(Sample {
            t: self.t,
            v: self.v,
            u: self.u,
        });

        while self.samples.len() > self.window_samples {
            self.samples.pop_front();
            self.euler_samples.pop_front();
        }

        let window_start = self.samples.front().map(|sample| sample.t).unwrap_or(self.t);
        while self.spike_times.front().is_some_and(|spike_time| *spike_time < window_start) {
            self.spike_times.pop_front();
        }
    }
}

fn slider<T>(ui: &mut egui::Ui, label: &str, value: &mut T, min: T, max: T) -> bool
where
    T: egui::emath::Numeric,
{
    ui.add(egui::Slider::new(value, min..=max).text(label)).changed()
}

fn euler_step(v: f32, u: f32, params: &NeuronParams, dt: f32) -> (f32, f32) {
    let (dv, du) = derivatives(v, u, params);

    let next_v = v + dv * dt;
    let next_u = u + du * dt;

    (next_v, next_u)
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
