use eframe::egui;
use egui_plot::{Legend, Line, Plot, Points, PlotPoints};
use std::time::Duration;
use image::load_from_memory;

mod qmath;
mod izh;

use serde::{Serialize, Deserialize};
use izh::{NeuronParams, LiveSimulation};

fn main() -> eframe::Result<()> {

    let icon_bytes = include_bytes!("../assets/icon.png");
    let icon_image = load_from_memory(icon_bytes).expect("Failed to load icon image").to_rgba8();

    let options = eframe::NativeOptions{

        viewport: egui::ViewportBuilder::default().with_icon(
            egui::IconData{
                rgba: icon_image.into_raw(),
                width: 512,
                height: 512,
            },
        ),
        ..Default::default()
    };
    eframe::run_native(
        "Izhikevich Neuron Visualizer",
        options,
        Box::new(|cc| Ok(Box::new(NeuronApp::new(cc)))),
    )
}



#[derive(Serialize, Deserialize, Clone)]
struct NeuronApp {
    params: NeuronParams,
    simulation: LiveSimulation,
    running: bool,
    plot_height: Option<f32>,
    target_fps: f32,
    steps_per_frame: u32,
    show_points: bool,
}

impl NeuronApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let neuron_params = cc.storage
            .and_then(|storage| eframe::get_value(storage, eframe::APP_KEY))
            .unwrap_or_default();

        let simulation = LiveSimulation::new(&neuron_params);

        Self { 
            params: neuron_params,
            simulation: simulation,
            ..Default::default()
         }
    }
}

impl Default for NeuronApp {
    fn default() -> Self {
        let params = NeuronParams::default();
        let simulation = LiveSimulation::new(&params);

        Self {
            params,
            simulation,
            running: true,
            plot_height: Some(500.0),
            target_fps: 30.0,
            steps_per_frame: 4,
            show_points: false,
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
                ui.label("Adjust the model parameters and watch the simulation evolve in real time.");

                ui.separator();

                ui.heading("Neural representation");

                egui::ComboBox::from_label("Model Type")
                    .selected_text(if let izh::NeuralModel::FixedPoint { bit_width, q_width } = self.params.model {
                        format!("Fixed Point Q{}.{}", bit_width, q_width).to_string()
                    } else {
                        "Floating Point".to_string()
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.params.model,
                            izh::NeuralModel::FloatingPoint,
                            "Floating Point",
                        );
                        ui.selectable_value(
                            &mut self.params.model,
                            izh::NeuralModel::FixedPoint { bit_width: 32, q_width: 16 },
                            "Fixed Point (32-bit, Q16)",
                        );
                    });

                match &mut self.params.model {
                    izh::NeuralModel::FloatingPoint => {
                        ui.label("Using 32-bit floating-point arithmetic for simulation.");
                    }
                    &mut izh::NeuralModel::FixedPoint { ref mut bit_width, ref mut q_width } => {
                        slider(ui, "Bit Width", bit_width, 8_usize, 32_usize);
                        slider(ui, "Q Width", q_width, 1_usize, *bit_width - 4);
                        ui.label(format!("Using fixed-point arithmetic with bit width {} and Q width {}.", bit_width, q_width));
                        ui.label(format!("Largest integer representable: {}", (1 << (*bit_width - *q_width - 1)) - 1));
                        ui.label(format!("Least significant bit value: {}", 1.0 / (1 << *q_width) as f64));
                    }
                }

                ui.separator();

                ui.heading("Neural Parameters");

                slider(ui, "a", &mut self.params.a, 0.001, 0.2);
                slider(ui, "b", &mut self.params.b, 0.01, 0.5);
                slider(ui, "c", &mut self.params.c, -80.0, -40.0);
                slider(ui, "d", &mut self.params.d, 0.0, 20.0);
                slider(ui, "dt (ms)", &mut self.params.dt, 0.01, 2.0);
                if slider(ui, "Number of neurons:", &mut self.params.num_neurons, 1, 10) {
                    needs_reset = true;
                }

                ui.label("View controls:");
                if slider(ui, "Window (ms)", &mut self.params.duration, 20.0, 1000.0) {
                    self.simulation.window_samples = ((self.params.duration / self.params.dt.max(0.001)).ceil() as usize).max(2) + 1;
                }
                slider(ui, "Update FPS", &mut self.target_fps, 1.0, 120.0);
                slider(ui, "Steps / frame", &mut self.steps_per_frame, 1, 25);

                ui.checkbox(&mut self.show_points, "Show points");

                ui.separator();

                ui.heading("Input Stimuli");

                if self.params.input_currents.len() != self.params.num_neurons {
                    self.params
                        .input_currents
                        .resize(self.params.num_neurons, 10.0);
                }
                
                for (neuron, neuron_current) in self.params.input_currents.iter_mut().enumerate() {
                    slider(ui, &format!("Neuron {} input current", neuron + 1),
                        neuron_current, 0.0, 50.0);
                }

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

                ui.label(format!("Live time: {:.2} ms", self.simulation.simulation_time));                
            });

        if needs_reset {
            self.simulation = LiveSimulation::new(&self.params);
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

            let voltage_traces = Vec::from_iter(
                self.simulation
                    .histories
                    .iter()
                    .map(|history| PlotPoints::from_iter(
                        history.iter().map(|sample| [sample.t as f64, sample.v as f64])
                    ))
            );

            // 2. Resolve or fallback the dynamic plot height state
            let mut current_height = self.plot_height.unwrap_or(260.0);

            Plot::new("voltage_plot")
                .height(current_height)
                .legend(Legend::default())
                .link_axis(shared_axis_group, shared_x_axes)
                .y_axis_label("voltage (mV)")
                .show(ui, |plot_ui| {
                    if self.show_points {
                        for (i, voltage_trace) in voltage_traces.into_iter().enumerate() {
                            plot_ui.points(Points::new(format!("Neuron {}", i + 1), voltage_trace).radius(2.0));
                        }; // Add points to the plot
                    } else {
                        for (i, voltage_trace) in voltage_traces.into_iter().enumerate() {
                            plot_ui.line(Line::new(format!("Neuron {}", i + 1), voltage_trace));
                        }; // Add lines to the plot
                    }
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

            let recovery_traces = Vec::from_iter(
                self.simulation
                    .histories
                    .iter()
                    .map(|history| PlotPoints::from_iter(
                        history.iter().map(|sample| [sample.t as f64, sample.u as f64])
                    ))
            );

            Plot::new("recovery_plot")
                .legend(Legend::default())
                .link_axis(shared_axis_group, shared_x_axes)
                .x_axis_label("time (ms)")
                .y_axis_label("u (a.u.)")
                .show(ui, |plot_ui| {
                    if self.show_points {
                        for (i, recovery_trace) in recovery_traces.into_iter().enumerate() {
                            plot_ui.points(Points::new(format!("Neuron {}", i + 1), recovery_trace).radius(2.0));
                        }; // Add points to the plot
                    } else {
                        for (i, recovery_trace) in recovery_traces.into_iter().enumerate() {
                            plot_ui.line(Line::new(format!("Neuron {}", i + 1), recovery_trace));
                        }; // Add lines to the plot
                    }
                });
        });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.params);
    }
}

fn slider<T>(ui: &mut egui::Ui, label: &str, value: &mut T, min: T, max: T) -> bool
where
    T: egui::emath::Numeric,
{
    ui.add(egui::Slider::new(value, min..=max).text(label)).changed()
}