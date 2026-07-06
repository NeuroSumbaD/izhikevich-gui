use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use std::collections::VecDeque;
use std::time::Duration;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
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
    spike_times: VecDeque<f32>,
}

struct NeuronApp {
    params: NeuronParams,
    simulation: LiveSimulation,
    running: bool,
    plot_height: Option<f32>,
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
            plot_height: Some(260.0),
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

            let shared_x_axes = egui::Vec2b::new(true, false);
            let shared_axis_group = "neuron_time_axis";

            ui.set_width(ui.available_width());

            ui.vertical_centered(|ui| {
                ui.heading("Membrane Potential");
            });

            let voltage_points = PlotPoints::from_iter(
                self.simulation
                    .samples
                    .iter()
                    .map(|sample| [sample.t as f64, sample.v as f64]),
            );

            // 2. Resolve or fallback the dynamic plot height state
            let mut current_height = self.plot_height.unwrap_or(260.0);

            Plot::new("voltage_plot")
                .height(current_height)
                .legend(Legend::default())
                .link_axis(shared_axis_group, shared_x_axes)
                .y_axis_label("voltage (mV)")
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Membrane Potential", voltage_points));
                });
                    // });

            // 4. Create an elegant, full-width custom drag divider handle directly below the plot
            let id = ui.id().with("plot_drag_handle");
            let slider_rect = ui.allocate_space(egui::vec2(ui.available_width(), 6.0)).1; // 6px tall hit box
            let response = ui.interact(slider_rect, id, egui::Sense::drag());

            // 5. Update state by capturing mouse drag changes on the Y-axis
            if response.dragged() {
                current_height += response.drag_delta().y;
                // Restrict minimum/maximum height limits safely
                current_height = current_height.clamp(100.0, 800.0); 
                self.plot_height = Some(current_height);
            }

            // 6. Visual cue: Draw a subtle hover/drag line indicator
            let visual_color = if response.dragged() {
                ui.visuals().widgets.active.bg_fill
            } else if response.hovered() {
                ui.set_cursor_icon(egui::CursorIcon::ResizeVertical);
                ui.visuals().widgets.hovered.bg_fill
            } else {
                ui.visuals().widgets.noninteractive.bg_fill.linear_multiply(0.5)
            };

            ui.painter().rect_filled(
                slider_rect.shrink2(egui::vec2(0.0, 2.0)), // Make the visible line 2px tall inside the 6px hit box
                2.0,
                visual_color,
            );

            ui.add_space(12.0);

            ui.vertical_centered(|ui| {
                ui.heading("Recovery variable");
            });

            let recovery_points = PlotPoints::from_iter(
                self.simulation
                    .samples
                    .iter()
                    .map(|sample| [sample.t as f64, sample.u as f64]),
            );

            Plot::new("recovery_plot")
                .legend(Legend::default())
                .link_axis(shared_axis_group, shared_x_axes)
                .x_axis_label("time (ms)")
                .y_axis_label("u (a.u.)")
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Recovery variable", recovery_points));
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
            spike_times: VecDeque::new(),
        };

        simulation.samples.push_back(Sample { t: 0.0, v, u });

        while simulation.t < params.duration {
            simulation.step(params);
        }

        simulation
    }

    fn step(&mut self, params: &NeuronParams) {
        let dt = params.dt.max(0.001);
        let (next_v, next_u) = rk4_step(self.v, self.u, params, dt);

        self.t += dt;
        self.v = next_v;
        self.u = next_u;

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
