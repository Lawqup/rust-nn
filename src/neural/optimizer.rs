use std::sync::mpsc::{self, Sender};

use crate::{
    matrix::Matrix2,
    neural::NeuralNet,
    prelude::*,
    viz::{EpochState, Visualizer},
};

use super::utils::Grad;

pub enum OptimizerMethod {
    Backprop,
}

pub struct Optimizer {
    method: OptimizerMethod,
    epochs: usize,
    epochs_per_log: Option<usize>,
    rate: f64,
    batch_size: Option<usize>,
}

impl Optimizer {
    pub fn new(method: OptimizerMethod, epochs: usize, rate: f64) -> Self {
        Self {
            method,
            epochs,
            epochs_per_log: None,
            rate,
            batch_size: None,
        }
    }

    pub fn with_batches(mut self, batch_size: Option<usize>) -> Self {
        self.batch_size = batch_size;
        self
    }
    pub fn with_log(mut self, iterations_per_log: Option<usize>) -> Self {
        self.epochs_per_log = iterations_per_log;
        self
    }

    pub fn set_rate(&mut self, rate: f64) {
        self.rate = rate;
    }

    pub fn set_iterations(&mut self, iterations: usize) {
        self.epochs = iterations;
    }

    fn train_internal(
        &self,
        net: &mut NeuralNet,
        inputs: &Matrix2<f64>,
        targets: &Matrix2<f64>,
        gui: bool,
        tx: Option<&Sender<EpochState>>,
    ) -> Result<()> {
        let mut inputs = inputs.clone();
        let mut targets = targets.clone();

        // Actually max batch size, last batch might be smaller
        let batch_size = self.batch_size.unwrap_or(inputs.rows());
        let n_batches = (inputs.rows() as f64 / batch_size as f64).ceil() as usize;

        let mut grad = Grad::empty(net, batch_size);
        for epoch in 0..self.epochs {
            // No need for SGD if input isn't batched
            if n_batches > 1 {
                Matrix2::shuffle_rows_synced(&mut inputs, &mut targets)?;
            }

            let mut error_sum = 0.0;
            for batch in 0..n_batches {
                let curr_batch = inputs.copy_rows(batch, batch_size);
                let curr_batch_targets = targets.copy_rows(batch, batch_size);
                match self.method {
                    OptimizerMethod::Backprop => {
                        self.backprop_once(net, &curr_batch, &curr_batch_targets, &mut grad)?
                    }
                }
                error_sum += net.mean_squared_error(&curr_batch, &curr_batch_targets)?;
            }
            if self.epochs_per_log.is_some_and(|ipl| epoch % ipl == 0) {
                let mse = error_sum / n_batches as f64;
                if gui {
                    tx.unwrap()
                        .send((epoch, mse, net.clone()))
                        .map_err(|_| Error::ThreadErr)?;
                } else {
                    println!("Epoch {epoch} error: {mse}")
                }
            }
        }
        Ok(())
    }

    pub fn train(
        &self,
        net: &mut NeuralNet,
        inputs: &Matrix2<f64>,
        targets: &Matrix2<f64>,
    ) -> Result<()> {
        self.train_internal(net, inputs, targets, false, None)
    }

    pub fn train_gui<Gui: Visualizer + 'static>(
        &self,
        net: &mut NeuralNet,
        inputs: &Matrix2<f64>,
        targets: &Matrix2<f64>,
    ) -> Result<()> {
        std::thread::scope(|scope| -> Result<()> {
            let (tx, rx) = mpsc::channel();
            let handle = scope.spawn(move || -> Result<()> {
                self.train_internal(net, inputs, targets, true, Some(&tx))
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
            d_acts[net.layers.len()].apply(|x| x * 2.0);

            for (l, layer) in net.layers.iter().enumerate().rev() {
                w_grads[i][l].zero();
                b_grads[i][l].zero();
                d_acts[l].zero();

                for n in 0..layer.weights.rows() {
                    let db =
                        d_acts[l + 1][(0, n)] * layer.activation.derivative(acts[l + 1][(0, n)]);
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
