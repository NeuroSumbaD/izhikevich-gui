use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};

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

struct SimulationResult {
    samples: Vec<Sample>,
    spike_times: Vec<f32>,
}

struct NeuronApp {
    params: NeuronParams,
    simulation: SimulationResult,
}

impl Default for NeuronApp {
    fn default() -> Self {
        let params = NeuronParams::default();
        let simulation = simulate(&params);

        Self { params, simulation }
    }
}

impl eframe::App for NeuronApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let mut needs_resimulate = false;

        egui::Panel::left("controls")
            .resizable(false)
            .default_size(280.0)
            .show_inside(ui, |ui| {
                ui.heading("Izhikevich Controls");
                ui.label("Adjust the model parameters and rerun the RK4 simulation.");

                ui.separator();

                needs_resimulate |= slider(ui, "a", &mut self.params.a, 0.001, 0.2);
                needs_resimulate |= slider(ui, "b", &mut self.params.b, 0.01, 0.5);
                needs_resimulate |= slider(ui, "c", &mut self.params.c, -80.0, -40.0);
                needs_resimulate |= slider(ui, "d", &mut self.params.d, 0.0, 20.0);
                needs_resimulate |= slider(ui, "dt (ms)", &mut self.params.dt, 0.01, 2.0);
                needs_resimulate |= slider(ui, "Input current", &mut self.params.input_current, 0.0, 50.0);
                needs_resimulate |= slider(ui, "Duration (ms)", &mut self.params.duration, 20.0, 1000.0);

                ui.separator();

                if ui.button("Run simulation").clicked() {
                    needs_resimulate = true;
                }

                ui.label(format!("Samples: {}", self.simulation.samples.len()));
                ui.label(format!("Spikes: {}", self.simulation.spike_times.len()));

                if let Some(last_spike) = self.simulation.spike_times.last() {
                    ui.label(format!("Last spike: {:.2} ms", last_spike));
                } else {
                    ui.label("Last spike: none");
                }

                if let Some(last) = self.simulation.samples.last() {
                    ui.label(format!("Final V: {:.2} mV", last.v));
                    ui.label(format!("Final u: {:.2}", last.u));
                }
            });

        if needs_resimulate {
            self.simulation = simulate(&self.params);
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

            Plot::new("voltage_plot")
                .legend(Legend::default())
                .height(260.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Membrane voltage (mV)", voltage_points));
                });

            ui.add_space(12.0);

            Plot::new("recovery_plot")
                .legend(Legend::default())
                .height(220.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Recovery variable u", recovery_points));
                });
        });
    }
}

fn slider(ui: &mut egui::Ui, label: &str, value: &mut f32, min: f32, max: f32) -> bool {
    ui.add(egui::Slider::new(value, min..=max).text(label)).changed()
}

fn simulate(params: &NeuronParams) -> SimulationResult {
    let mut samples = Vec::new();
    let mut spike_times = Vec::new();

    let mut t = 0.0_f32;
    let mut v = -65.0_f32;
    let mut u = params.b * v;

    samples.push(Sample { t, v, u });

    let dt = params.dt.max(0.001);
    let total_steps = (params.duration / dt).ceil() as usize;

    for _ in 0..total_steps {
        let (next_v, next_u) = rk4_step(v, u, params, dt);
        t += dt;

        v = next_v;
        u = next_u;

        if v >= 30.0 {
            samples.push(Sample { t, v: 30.0, u });
            spike_times.push(t);
            v = params.c;
            u += params.d;
        }

        samples.push(Sample { t, v, u });
    }

    SimulationResult { samples, spike_times }
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
