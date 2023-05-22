use crate::{
    matrix::{Matrix1, Matrix2},
    neural::{NNError, NeuralNet},
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
    rate: f64,
}

impl Optimizer {
    pub fn new(method: OptimizerMethod, iterations: usize, rate: f64) -> Self {
        Self {
            method,
            iterations,
            iterations_per_log: None,
            rate,
        }
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
    ) -> Result<(), NNError> {
        for i in 0..self.iterations {
            match self.method {
                OptimizerMethod::FiniteDiff(eps) => {
                    self.finite_diff_once(net, eps, inputs, targets)?
                }
                OptimizerMethod::Backprop => self.backprop_once(net, inputs, targets)?,
            }
            if self.iterations_per_log.is_some_and(|ipl| i % ipl == 0) {
                println!(
                    "Iteration {i} error: {}",
                    net.mean_squared_error(inputs, targets)?
                )
            }
        }
        Ok(())
    }

    fn finite_diff_once(
        &self,
        net: &mut NeuralNet,
        eps: f64,
        inputs: &Matrix2<f64>,
        targets: &Matrix2<f64>,
    ) -> Result<(), NNError> {
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
    ) -> Result<(), NNError> {
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

            d_acts[net.layers.len()] =
                (&acts[acts.len() - 1] - &targets[i]).map_err(|_| NNError::InputErr)?;

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

                layer.biases = (&layer.biases + &b_grads[i][l]).map_err(|_| NNError::InputErr)?;
                layer.weights = (&layer.weights + &w_grads[i][l]).map_err(|_| NNError::InputErr)?;
            }
        }

        Ok(())
    }
}