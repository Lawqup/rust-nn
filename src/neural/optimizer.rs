use std::sync::mpsc::{self};

use crate::{
    matrix::{Matrix1, Matrix2},
    neural::NeuralNet,
    prelude::*,
    viz::App,
};

pub enum OptimizerMethod {
    /// Takes in an epsilon used for approximating derivatives
    FiniteDiff(f64),
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
        for i in 0..self.iterations {
            match self.method {
                OptimizerMethod::FiniteDiff(eps) => {
                    self.finite_diff_once(net, eps, inputs, targets)?
                }
                OptimizerMethod::Backprop => self.backprop_once(net, inputs, targets)?,
            }
            if self.iterations_per_log.is_some_and(|ipl| i % ipl == 0) {
                let mse = net.mean_squared_error(inputs, targets)?;
                println!("Iteration {i} error: {mse}")
            }
        }
        Ok(())
    }

    pub fn train_gui(
        &self,
        net: &mut NeuralNet,
        inputs: &Matrix2<f64>,
        targets: &Matrix2<f64>,
    ) -> Result<()> {
        std::thread::scope(|scope| -> Result<()> {
            let (tx, rx) = mpsc::channel();
            let handle = scope.spawn(move || -> Result<()> {
                for i in 0..self.iterations {
                    match self.method {
                        OptimizerMethod::FiniteDiff(eps) => {
                            self.finite_diff_once(net, eps, inputs, targets)?
                        }
                        OptimizerMethod::Backprop => self.backprop_once(net, inputs, targets)?,
                    }
                    if self.iterations_per_log.is_some_and(|ipl| i % ipl == 0) {
                        let mse = net.mean_squared_error(inputs, targets)?;
                        tx.send([i as f64, mse]).map_err(|_| Error::ThreadErr)?;
                    }
                }
                Ok(())
            });

            let _ = eframe::run_native(
                "RustNN",
                eframe::NativeOptions::default(),
                Box::new(|cc| Box::new(App::new(cc, rx))),
            );

            handle.join().map_err(|_| Error::ThreadErr)??;
            Ok(())
        })
    }

    fn finite_diff_once(
        &self,
        net: &mut NeuralNet,
        eps: f64,
        inputs: &Matrix2<f64>,
        targets: &Matrix2<f64>,
    ) -> Result<()> {
        let mut w_grads = Vec::new();
        let mut b_grads = Vec::new();

        let costs = net.mean_squared_error(inputs, targets)?;
        for i in 0..net.layers.len() {
            w_grads.push(net.layers[i].weights.clone());
            b_grads.push(net.layers[i].biases.clone());

            for j in 0..net.layers[i].weights.rows() {
                for k in 0..net.layers[i].weights.cols() {
                    let saved = net.layers[i].weights[j][k];
                    net.layers[i].weights[j][k] += eps;
                    let dw = (net.mean_squared_error(inputs, targets)? - costs) / eps;

                    net.layers[i].weights[j][k] = saved;
                    w_grads[i][j][k] = -self.rate * dw;
                }

                let saved = net.layers[i].biases[j];
                net.layers[i].biases[j] += eps;
                let db = (net.mean_squared_error(inputs, targets)? - costs) / eps;

                net.layers[i].biases[j] = saved;
                b_grads[i][j] = -self.rate * db;
            }
        }

        for i in 0..w_grads.len() {
            net.layers[i].weights = (&net.layers[i].weights + &w_grads[i]).unwrap();
            net.layers[i].biases = (&net.layers[i].biases + &b_grads[i]).unwrap();
        }
        Ok(())
    }

    fn backprop_once(
        &self,
        net: &mut NeuralNet,
        inputs: &Matrix2<f64>,
        targets: &Matrix2<f64>,
    ) -> Result<()> {
        let mut w_grads = Vec::new();
        let mut b_grads = Vec::new();

        // i -- current input sample
        // l -- current layer
        // n -- current neuron
        // p -- previous neuron
        for (i, input) in inputs.iter().enumerate() {
            let mut w_grad = vec![Matrix2::<f64>::new(0, 0); net.layers.len()];
            let mut b_grad = vec![Matrix1::<f64>::new(0); net.layers.len()];

            let acts = net.forward(input)?;
            let mut d_acts = vec![Matrix1::new(0); acts.len()];

            d_acts[net.layers.len()] = (&acts[acts.len() - 1] - &targets[i])?;

            for (l, layer) in net.layers.iter().enumerate().rev() {
                d_acts[l] = Matrix1::new(layer.weights.cols());
                w_grad[l] = Matrix2::new(layer.weights.rows(), layer.weights.cols());
                b_grad[l] = Matrix1::new(layer.biases.size());

                for n in 0..layer.weights.rows() {
                    let db = 2.0 * d_acts[l + 1][n] * layer.activation.derivative(acts[l + 1][n]);
                    b_grad[l][n] += db;
                    for p in 0..layer.weights.cols() {
                        w_grad[l][n][p] += db * acts[l][p];
                        d_acts[l][p] += db * layer.weights[n][p];
                    }
                }
            }

            w_grads.push(w_grad);
            b_grads.push(b_grad);
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
