# Izhikevich Neuron Visualizer
This repository is a small interactive simulator for Izhikevich neuron models, with an emphasis on limited-precision arithmetic and fixed-point formats that resemble what you would implement in digital logic.

[Click here](https://neurosumbad.github.io/izhikevich-gui/) to run the application directly in your browser.

![Description for screen readers](/assets/Screenshot.png)
*Figure 1: Sample screenshot.*


It is built as a desktop GUI in Rust using `eframe` and `egui_plot`, so you can tune model parameters, compare floating-point and fixed-point behavior, and watch the voltage and recovery-variable traces update in real time.

## What it demonstrates

- A live Izhikevich neuron simulation with configurable time step, duration, and input current.
- Floating-point execution for reference behavior.
- Fixed-point execution with user-selectable bit width and Q width.
- Per-neuron input currents for simulating a small group of neurons at once.
- Persistent parameter storage so the app can reopen with the last saved control values.

## Why the fixed-point path matters

The code is designed to explore the tradeoff between numerical fidelity and hardware-friendly arithmetic:

- `src/qmath.rs` defines a small `FixedPoint` type with explicit bit width and Q width.
- Arithmetic operators are overloaded so the fixed-point code stays readable.
- Multiplication and addition expand the intermediate width, then results are truncated back down to the target format.
- Values are clamped when constructed so the simulation stays within the representable range.

That makes the project useful as a quick testbed for digital-logic-oriented neuron models, where the question is not just whether the dynamics work, but how much precision is needed for stable behavior.

## Architecture

- `src/main.rs` contains the GUI, persistence, parameter editing, and plotting.
- `src/izh.rs` contains the neuron model, simulation state, and the floating-point / fixed-point stepping code.
- `src/qmath.rs` contains the fixed-point implementation used by the simulation.

The simulation keeps a short rolling history for each neuron so the plots stay responsive while the app is running.

## Running

Build and run with Cargo:

```bash
cargo run --release
```

Using `--release` is recommended because the simulation steps are executed continuously while the UI is open.

## Controls

The left panel lets you adjust:

- Model type: floating point or fixed point.
- Fixed-point format: bit width and Q width.
- Izhikevich parameters: `a`, `b`, `c`, `d`.
- Simulation timing: `dt`, window length, update FPS, and steps per frame.
- Number of neurons and their individual input currents.

The right side plots:

- Membrane potential `v` for each neuron.
- Recovery variable `u` for each neuron.

## Intended scope

This is not a general-purpose neural simulator or a large-scale network engine. It is meant to be a compact, practical tool for exploring a handful of neurons and comparing precision formats before moving to a hardware-oriented implementation (such as in an FPGA). Currently, the simulation is limited to 10 neurons, otherwise the plots become too hard to interpret, but I may switch to raster plots in the future if there is a need. Use this tool if you want to fine-tune the neural parameters in real-time to see when they give the dynamics you expect.

### *Notes*:
 1. It seems that with fixed representation width, Q8.9 is the smallest working representation with the traditional Izhikevich parameters

## Upcoming features
 - Inhibitory competition between neurons
 - Separate window to plot the firing rate vs input
 - Saving simulations as PNG or CSV data