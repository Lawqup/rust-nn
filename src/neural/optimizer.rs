use std::sync::mpsc;

use crate::{matrix::Matrix2, neural::NeuralNet, prelude::*, viz::Visualizer};

use super::utils::Grad;

pub enum OptimizerMethod {
    Backprop,
}

pub struct Optimizer {
    method: OptimizerMethod,
    iterations: usize,
    iterations_per_log: Option<usize>,
    gui: bool,
    rate: f64,
}

impl Optimizer {
    pub fn new(method: OptimizerMethod, iterations: usize, rate: f64) -> Self {
        Self {
            method,
            iterations,
            iterations_per_log: None,
            gui: false,
            rate,
        }
    }

    pub fn toggle_gui(mut self) -> Self {
        self.gui = !self.gui;
        self
    }
    pub fn with_log(mut self, iterations_per_log: Option<usize>) -> Self {
        self.iterations_per_log = iterations_per_log;
        self
    }

    pub fn set_rate(&mut self, rate: f64) {
        self.rate = rate;
    }

    pub fn set_iterations(&mut self, iterations: usize) {
        self.iterations = iterations;
    }

    pub fn train(
        &self,
        net: &mut NeuralNet,
        inputs: &Matrix2<f64>,
        targets: &Matrix2<f64>,
    ) -> Result<()> {
        let mut grad = Grad::empty(net, inputs.rows());
        for i in 0..self.iterations {
            match self.method {
                OptimizerMethod::Backprop => self.backprop_once(net, inputs, targets, &mut grad)?,
            }
            if self.iterations_per_log.is_some_and(|ipl| i % ipl == 0) {
                let mse = net.mean_squared_error(inputs, targets)?;
                println!("Iteration {i} error: {mse}")
            }
        }
        Ok(())
    }

    pub fn train_gui<Gui: Visualizer>(
        &self,
        net: &mut NeuralNet,
        inputs: &Matrix2<f64>,
        targets: &Matrix2<f64>,
    ) -> Result<()> {
        let mut grad = Grad::empty(net, inputs.rows());

        std::thread::scope(|scope| -> Result<()> {
            let (tx, rx) = mpsc::channel();
            let handle = scope.spawn(move || -> Result<()> {
                for i in 0..self.iterations {
                    match self.method {
                        OptimizerMethod::Backprop => {
                            self.backprop_once(net, inputs, targets, &mut grad)?
                        }
                    }
                    if self.iterations_per_log.is_some_and(|ipl| i % ipl == 0) {
                        let mse = net.mean_squared_error(inputs, targets)?;
                        let outputs = net.run_batch(inputs)?;
                        tx.send((i, mse, outputs)).map_err(|_| Error::ThreadErr)?;
                    }
                }
                Ok(())
            });

            let _ = eframe::run_native(
                "RustNN",
                eframe::NativeOptions::default(),
                Box::new(|cc| Box::new(Gui::new(cc, rx))),
            );

            handle.join().map_err(|_| Error::ThreadErr)??;
            Ok(())
        })
    }

    fn backprop_once(
        &self,
        net: &mut NeuralNet,
        inputs: &Matrix2<f64>,
        targets: &Matrix2<f64>,
        Grad(w_grads, b_grads, d_acts): &mut Grad,
    ) -> Result<()> {
        // i -- current input sample
        // l -- current layer
        // n -- current neuron
        // p -- previous neuron
        for i in 0..inputs.rows() {
            let acts = net.forward(&inputs.clone_row(i))?;

            d_acts[net.layers.len()] = (&acts[acts.len() - 1] - &targets.clone_row(i))?;

            for (l, layer) in net.layers.iter().enumerate().rev() {
                w_grads[i][l].zero();
                b_grads[i][l].zero();
                d_acts[l].zero();

                for n in 0..layer.weights.rows() {
                    let db = 2.0
                        * d_acts[l + 1][(0, n)]
                        * layer.activation.derivative(acts[l + 1][(0, n)]);
                    b_grads[i][l][(0, n)] += db;
                    for p in 0..layer.weights.cols() {
                        w_grads[i][l][(n, p)] += db * acts[l][(0, p)];
                        d_acts[l][(0, p)] += db * layer.weights[(n, p)];
                    }
                }
            }
        }

        for i in 0..inputs.rows() {
            for (l, layer) in net.layers.iter_mut().enumerate() {
                b_grads[i][l].apply(|x| -self.rate * x / inputs.rows() as f64);
                w_grads[i][l].apply(|x| -self.rate * x / inputs.rows() as f64);

                layer.biases = (&layer.biases + &b_grads[i][l])?;
                layer.weights = (&layer.weights + &w_grads[i][l])?;
            }
        }

        Ok(())
    }
}
