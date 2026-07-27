use eframe::egui;
use egui_plot::{Legend, Line, Plot, Points, PlotPoints};
use std::time::Duration;
use image::load_from_memory;
use rand::{SeedableRng, rngs::SmallRng};

mod qmath;
mod izh;
use izh::{NeuronParams, LiveSimulation, RateEstimateMethod};

fn main() -> eframe::Result<()> {

    let icon_bytes = include_bytes!("../assets/icon.png");
    let icon_image = load_from_memory(icon_bytes).expect("Failed to load icon image").to_rgba8();

    let options = eframe::NativeOptions{
        viewport: egui::ViewportBuilder::default()
            .with_icon(
                egui::IconData{
                    rgba: icon_image.into_raw(),
                    width: 512,
                    height: 512,
                },
            )
            .with_min_inner_size([600.0, 400.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Izhikevich Neuron Visualizer",
        options,
        Box::new(|cc| Ok(Box::new(NeuronApp::new(cc)))),
    )
}



struct NeuronApp {
    params: NeuronParams,
    simulation: LiveSimulation,
    running: bool,
    plot_height: Option<f32>,
    target_fps: f32,
    steps_per_frame: u32,
    show_points: bool,
    rng: SmallRng,
    show_filtered: bool,
    show_stimuli: bool,
    max_expected_rate: f64,
}

impl NeuronApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let neuron_params: NeuronParams = cc.storage
            .and_then(|storage| eframe::get_value(storage, eframe::APP_KEY))
            .unwrap_or_default();

        let mut rng = SmallRng::from_seed(neuron_params.rng_seed);
        let simulation = LiveSimulation::new(&neuron_params, &mut rng);

        Self { 
            params: neuron_params,
            simulation: simulation,
            rng: rng,
            ..Default::default()
         }
    }
}

impl Default for NeuronApp {
    fn default() -> Self {
        let params = NeuronParams::default();
        let mut rng = SmallRng::from_seed(params.rng_seed);
        let simulation = LiveSimulation::new(&params, &mut rng);

        Self {
            params,
            simulation,
            running: true,
            plot_height: Some(500.0),
            target_fps: 60.0,
            steps_per_frame: 4,
            show_points: false,
            rng: rng,
            show_filtered: false,
            show_stimuli: false,
            max_expected_rate: 100.0,
        }
    }
}

impl eframe::App for NeuronApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let mut needs_reset = false;

        let shared_x_axes = egui::Vec2b::new(true, false);
        let shared_axis_group = "neuron_time_axis";

        egui::Panel::left("controls")
            // .resizable(false)
            .default_size(280.0)
            .show_inside(ui, |ui| {
                ui.heading("Izhikevich Controls");
                ui.label("Adjust the model parameters and watch the simulation evolve in real time.");

                ui.separator();

                egui::CollapsingHeader::new(egui::RichText::new("Neural representation").heading())
                    .default_open(true)
                    .show(ui, |ui| {
                        // Content for the collapsing header
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
                    });


                ui.separator();

                egui::CollapsingHeader::new(egui::RichText::new("Neural Parameters").heading())
                    .default_open(true)
                    .show(ui, |ui| {
                        
                        egui::CollapsingHeader::new("Izhikevich parameters:")
                            .default_open(true)
                            .show(ui, |ui| {
                                slider(ui, "a", &mut self.params.a, 0.001, 0.2);
                                slider(ui, "b", &mut self.params.b, 0.01, 0.5);
                                slider(ui, "c", &mut self.params.c, -80.0, -40.0);
                                slider(ui, "d", &mut self.params.d, 0.0, 20.0);
                                slider(ui, "dt (ms)", &mut self.params.dt, 0.01, 2.0);
                            });

                        egui::CollapsingHeader::new("Conductances:")
                            .default_open(true)
                            .show(ui, |ui| {
                                // Add your content here
                                slider(ui, "excitatory reversal potential (mV)", &mut self.params.rev_e, -45.0, 100.0);
                                slider(ui, "maximum excitatory conductance", &mut self.params.ge_bar, 0.0, 4.0);
                                slider(ui, "leakage reversal potential (mV)", &mut self.params.rev_l, -100.0, -45.0);
                                slider(ui, "leakage conductance", &mut self.params.gl_bar, 0.0, 4.0);
                                slider(ui, "inhibitory reversal potential (mV)", &mut self.params.rev_i, -100.0, -45.0);
                                slider(ui, "maximum inhibitory conductance", &mut self.params.gi_bar, 0.0, 4.0);
                            });
                            
                        egui::CollapsingHeader::new("Rate-code parameters:")
                            .default_open(true)
                            .show(ui, |ui| {
                                egui::ComboBox::from_label("Rate Estimation Method")
                                    .selected_text(format!("{:?}", self.params.rate_estimate_method))
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut self.params.rate_estimate_method, RateEstimateMethod::SpikeFilter, "Spike Filter");
                                        ui.selectable_value(&mut self.params.rate_estimate_method, RateEstimateMethod::Isi, "ISI");
                                    });

                                slider(ui, "Leaky rate time constant (ms)", &mut self.params.leaky_rate_tau, 1.0, 1000.0);
                            });

                        egui::CollapsingHeader::new("FFFB inhibition parameters:")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.checkbox(&mut self.params.use_fffb, "Use FFFB inhibition");
                                slider(ui, "max vs avg:", &mut self.params.fffb_params.max_vs_avg, 0.0, 1.0);
                                slider(ui, "feedforward component:", &mut self.params.fffb_params.ff, 0.0, 1.0);
                                slider(ui, "initial feedforward activity:", &mut self.params.fffb_params.ff0, 0.0, 1.0);
                                slider(ui, "feedback delay (ms):", &mut self.params.fffb_params.fb_dt, 0.0, 10.0);
                                slider(ui, "feedback component:", &mut self.params.fffb_params.fb, 0.0, 1.0);
                                slider(ui, "inhibitory conductance:", &mut self.params.fffb_params.gi, 0.0, 10.0); 
                            });

                        if slider(ui, "Number of neurons:", &mut self.params.num_neurons, 1, 10) {
                            needs_reset = true;
                        }
                    });

                egui::CollapsingHeader::new("View controls")
                    .default_open(true)
                    .show(ui, |ui| {
                        if slider(ui, "Window (ms)", &mut self.params.duration, 20.0, 1000.0) {
                            self.simulation.window_samples = ((self.params.duration / self.params.dt.max(0.001)).ceil() as usize).max(2) + 1;
                        }
                        slider(ui, "Update FPS", &mut self.target_fps, 1.0, 120.0);
                        slider(ui, "Steps / frame", &mut self.steps_per_frame, 1, 25);
        
                        ui.checkbox(&mut self.show_points, "Show points");
                    });


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

                    if ui.button("Stimuli Editor").clicked() {
                        self.show_stimuli = !self.show_stimuli;
                    }

                    if ui.button("show filtered traces").clicked() {
                        self.show_filtered = !self.show_filtered;
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

                let live_time = if self.simulation.simulation_time < 1000.0 {
                    format!("{:.2} ms", self.simulation.simulation_time)
                } else {
                    format!("{:.2} s", self.simulation.simulation_time / 1000.0)
                };
                ui.label(format!("Live time: {}", live_time));
            });

        if needs_reset {
            // Reset the rng with the same seed to ensure reproducibility
            self.rng = SmallRng::from_seed(self.params.rng_seed);
            self.simulation = LiveSimulation::new(&self.params, &mut self.rng);
        }

        if self.running {
            for _ in 0..self.steps_per_frame {
                self.simulation.step(&self.params, &mut self.rng);
            }
            ui.ctx()
                .request_repaint_after(Duration::from_secs_f32(1.0 / self.target_fps.max(1.0)));
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Neuron Response");
            ui.add_space(8.0);

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
                .label_formatter(|name, value| {
                    if !name.is_empty() {
                        format!("{}: {:.3}, {:.3}", name, value.x, value.y)
                    } else {
                        "".to_owned()
                    }
                })
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
                .label_formatter(|name, value| {
                    if !name.is_empty() {
                        format!("{}: {:.3}, {:.3}", name, value.x, value.y)
                    } else {
                        "".to_owned()
                    }
                })
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

        if self.show_stimuli {
            // Show stimuli editor
            egui::Window::new("Stimuli Editor")
                .open(&mut self.show_stimuli)
                .min_height(300.0)
                .show(ui.ctx(), |ui| {
                    ui.heading("Input Stimuli");

                    if self.params.exc_inputs.len() != self.params.num_neurons {
                        self.params
                            .exc_inputs
                            .resize(self.params.num_neurons, 0.1);

                        self.params
                            .noise_std_devs
                            .resize(self.params.num_neurons, 0.001);
                    }
                    
                    egui::ScrollArea::vertical()
                        // 2. Instruct the scroll area to fill the remaining panel height
                        .auto_shrink([false; 2]) 
                        .show(ui, |ui| {
                            for (neuron, neuron_current) in self.params.exc_inputs.iter_mut().enumerate() {
                                ui.label(format!("Neuron {}", neuron + 1));
                                slider(ui, &"Normalized Excitatory input", neuron_current, 0.0, 1.0);
                                slider(ui, &"Input Noise std dev", &mut self.params.noise_std_devs[neuron], 0.0, 0.2);
                            }
                            ui.add(egui::Separator::default().spacing(5.0));
                    });
                });
        }

        if self.show_filtered {
            // Retrieve rate-code traces from the simulation
            let filtered_traces = Vec::from_iter(
                self.simulation
                    .rate_histories
                    .iter()
                    .zip(self.simulation.histories.iter())
                    .map(|history| PlotPoints::from_iter(
                        history.0.iter().zip(history.1.iter()).map(|samples| [samples.1.t as f64, *samples.0 as f64])
                    ))
            );

            // Show filtered traces
            egui::Window::new("Filtered traces")
            .open(&mut self.show_filtered)
            .show(ui.ctx(), |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("Rate-coded traces");
                });

                slider(ui, "Max expected rate (Hz)", &mut self.max_expected_rate, 1.0, 1000.0);

                Plot::new("Filtered_plot")
                    .legend(Legend::default())
                    .label_formatter(|name, value| {
                        if !name.is_empty() {
                            format!("{}: {:.3}, {:.3}", name, value.x, value.y)
                        } else {
                            "".to_owned()
                        }
                    })
                    .link_axis(shared_axis_group, shared_x_axes)
                    .x_axis_label("time (ms)")
                    .y_axis_label("rate (Hz)")
                    .auto_bounds([true, false])
                    .show(ui, |plot_ui| {
                        plot_ui.set_plot_bounds_y(0.0_f64..=(self.max_expected_rate));
                        // draw filtered traces here
                        for (i, filtered_trace) in filtered_traces.into_iter().enumerate() {
                            plot_ui.line(Line::new(format!("Neuron {}", i + 1), filtered_trace));
                        }; // Add lines to the plot
                });
            });
        }
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