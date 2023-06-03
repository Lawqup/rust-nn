pub mod activations;
pub mod optimizer;
mod utils;

use crate::prelude::*;
use std::{ops::RangeInclusive, sync::Arc};

use crate::matrix::{
    ops::{Dot, Transpose},
    Matrix2,
};
use rand::distributions::{Distribution, Uniform};

use self::activations::{Activation, Activations};

#[derive(Clone)]
pub struct DenseLayer {
    weights: Matrix2<f64>,
    biases: Matrix2<f64>,
    activation: Arc<dyn Activation>,
}

#[derive(Clone)]
pub struct NeuralNet {
    layers: Vec<DenseLayer>,
}

impl DenseLayer {
    /// Initializes a layer given the number of inputs and neurons.
    /// Params initialized as a random value in init_range
    /// Activation function is initally the identity (f(x) = x)
    pub fn new(n_inputs: u32, n_neurons: u32, init_range: RangeInclusive<f64>) -> Self {
        let mut rng = rand::thread_rng();
        let die = Uniform::from(init_range);

        let weights = Matrix2::from_vec(
            (0..n_neurons)
                .map(|_| {
                    (0..n_inputs)
                        .map(|_| die.sample(&mut rng))
                        .collect::<Vec<_>>()
                })
                .collect(),
        )
        .unwrap();

        let biases = Matrix2::from_row((0..n_neurons).map(|_| die.sample(&mut rng)).collect());

        Self {
            weights,
            biases,
            activation: Arc::new(Activations::Identity),
        }
    }

    /// Add an activation function of this layer
    pub fn with_activation(mut self, activation: impl Activation + 'static) -> Self {
        self.activation = Arc::new(activation);
        self
    }

    /// Propogates a batch of inputs through the layer applying the activation function.
    pub fn run_batch(&self, input: &Matrix2<f64>) -> Result<Matrix2<f64>> {
        let mut res = input.dot(&self.weights.transpose())?;
        for row in 0..res.rows() {
            for col in 0..res.cols() {
                res[(row, col)] += self.biases[(0, col)]
            }
        }
        res.apply(|x| self.activation.call(x));
        Ok(res)
    }

    /// Propogates an input through the layer applying the activation function.
    pub fn forward(&self, input: &Matrix2<f64>) -> Result<Matrix2<f64>> {
        let mut res = input.dot(&self.weights.transpose())?;
        for col in 0..res.cols() {
            res[(0, col)] += self.biases[(0, col)]
        }
        res.apply(|x| self.activation.call(x));
        Ok(res)
    }

    /// Returns the amount of inputs this layer accepts
    pub fn input_amount(&self) -> u32 {
        self.weights.dim().1 as u32
    }

    /// Returns the amount of neurons in the layer
    pub fn neuron_amount(&self) -> u32 {
        self.weights.dim().0 as u32
    }
}

impl NeuralNet {
    /// Creates a 1-layer neural net with some number of neurons
    /// that can accept a certain number of inputs
    pub fn new(n_inputs: u32, n_neurons: u32, activation: impl Activation + 'static) -> Self {
        Self {
            layers: vec![
                DenseLayer::new(n_inputs, n_neurons, -1.0..=1.0).with_activation(activation)
            ],
        }
    }

    pub fn add_layer(&mut self, n_neurons: u32, activation: impl Activation + 'static) {
        let n_inputs = self.layers.last().unwrap().neuron_amount();
        self.layers
            .push(DenseLayer::new(n_inputs, n_neurons, -1.0..=1.0).with_activation(activation));
    }

    /// Reset parameters to uniformly random values between a specified range
    pub fn randomize(&mut self, r: RangeInclusive<f64>) {
        let mut rng = rand::thread_rng();
        let die = Uniform::from(r);

        for layer in self.layers.iter_mut() {
            for b in 0..layer.biases.cols() {
                layer.biases[(0, b)] = die.sample(&mut rng);
            }
            for row in 0..layer.weights.rows() {
                for col in 0..layer.weights.cols() {
                    layer.weights[(row, col)] = die.sample(&mut rng);
                }
            }
        }
    }

    pub fn forward(&self, input: &Matrix2<f64>) -> Result<Vec<Matrix2<f64>>> {
        // NN has to be initialized with at least one layer
        let first_layer = self.layers.first().unwrap();

        if input.cols() as u32 != first_layer.input_amount() {
            return Err(Error::DimensionErr);
        }

        let mut outputs = vec![input.clone()];
        for layer in self.layers.iter() {
            outputs.push(layer.forward(&outputs[outputs.len() - 1]).unwrap());
        }
        Ok(outputs)
    }

    /// Propogates a batch of inputs through layers
    pub fn run_batch(&self, inputs: &Matrix2<f64>) -> Result<Matrix2<f64>> {
        // NN has to be initialized with at least one layer
        let first_layer = self.layers.first().unwrap();

        if inputs.cols() as u32 != first_layer.input_amount() {
            return Err(Error::DimensionErr);
        }

        let mut prev_output = first_layer.run_batch(inputs).unwrap();
        for layer in self.layers.iter().skip(1) {
            prev_output = layer.run_batch(&prev_output).unwrap();
        }
        Ok(prev_output)
    }

    /// Mean-squared error
    pub fn mean_squared_error(&self, inputs: &Matrix2<f64>, targets: &Matrix2<f64>) -> Result<f64> {
        // Prediction rows must match class target size
        let outputs = self.run_batch(inputs)?;
        if outputs.dim() != targets.dim() {
            return Err(Error::DimensionErr);
        }

        let mut sum = 0.0;
        for row in 0..outputs.rows() {
            for col in 0..outputs.cols() {
                let diff = outputs[(row, col)] - targets[(row, col)];
                sum += diff * diff;
            }
        }
        Ok(sum / (targets.rows() * targets.cols()) as f64)
    }
}

#[cfg(test)]
mod tests {
    use crate::neural::optimizer::{Optimizer, OptimizerMethod};

    use super::*;

    fn default_inputs() -> Matrix2<f64> {
        Matrix2::from_array([
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8],
            [-1.5, 2.7, 3.3, -0.8],
        ])
    }

    #[test]
    fn dense_layer() {
        let layer1 = DenseLayer::new(4, 2, -1.0..=1.0);
        let layer2 = DenseLayer::new(2, 1, -3.0..=5.0);

        let layer1_out = layer1.run_batch(&default_inputs()).unwrap();

        let layer2_out = layer2.run_batch(&layer1_out).unwrap();

        assert_eq!(layer2_out.dim(), (4, 1));
        assert_eq!(layer2_out.as_vec()[2], layer2_out.as_vec()[3]);
    }

    #[test]
    fn activation_function() {
        let net = NeuralNet::new(4, 20, Activations::Sigmoid);

        let out = net.run_batch(&default_inputs()).unwrap();

        assert_eq!(out.dim(), (4, 20));
        assert!(out
            .to_vec()
            .into_iter()
            .all(|row| row.into_iter().all(|x| x < 1.0 && x > 0.0)));
    }

    #[test]
    fn randomize() {
        let mut net = NeuralNet::new(4, 20, Activations::Sigmoid);

        for layer in net.layers.iter() {
            for &b in &layer.biases.as_vec()[0] {
                assert!(-1.0 <= *b && *b <= 1.0);
            }
            for row in &layer.weights.as_vec() {
                for &w in row {
                    assert!(-1.0 <= *w && *w <= 1.0);
                }
            }
        }

        net.randomize(2.0..=5.0);

        for layer in net.layers.iter() {
            for &b in &layer.biases.as_vec()[0] {
                assert!(2.0 <= *b && *b <= 5.0);
            }
            for row in &layer.weights.as_vec() {
                for &w in row {
                    assert!(2.0 <= *w && *w <= 5.0);
                }
            }
        }
    }

    #[test]
    fn train_or() {
        // Train a single neuron to compute OR
        let mut net = NeuralNet::new(2, 1, Activations::Sigmoid);

        let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
        let targets = Matrix2::from_array([[0], [1], [1], [1]]).into();

        let rate = 10e-0;

        let optim = Optimizer::new(OptimizerMethod::Backprop, 1_000, rate);
        assert_eq!(Ok(()), optim.train(&mut net, &inputs, &targets));

        let fin = net.mean_squared_error(&inputs, &targets).unwrap();

        println!("------------------");
        println!("Final cost: {fin}");

        let res = net.run_batch(&inputs).unwrap();
        for (inp, out) in inputs.to_vec().into_iter().zip(res.to_vec()) {
            println!("{:?} -> {}", inp.to_vec(), out[0])
        }

        assert!(fin < 0.01);
    }

    #[test]
    fn train_xor() {
        let mut net = NeuralNet::new(2, 2, Activations::Arctan);
        net.add_layer(1, Activations::Sigmoid);

        let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
        let targets = Matrix2::from_array([[0], [1], [1], [0]]).into();

        let rate = 10e-0;

        let optim = Optimizer::new(OptimizerMethod::Backprop, 10_000, rate).with_log(Some(1_000));
        assert_eq!(Ok(()), optim.train(&mut net, &inputs, &targets));

        let fin = net.mean_squared_error(&inputs, &targets).unwrap();

        println!("------------------");
        println!("Final cost: {fin}");

        let res = net.run_batch(&inputs).unwrap();
        for (inp, out) in inputs.to_vec().into_iter().zip(res.to_vec()) {
            println!("{:?} -> {}", inp.to_vec(), out[0])
        }

        assert!(fin < 0.01);
    }

    #[test]
    fn train_xor_batched() {
        let mut net = NeuralNet::new(2, 2, Activations::Arctan);
        net.add_layer(1, Activations::Sigmoid);

        let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
        let targets = Matrix2::from_array([[0], [1], [1], [0]]).into();

        let rate = 10e-0;

        let optim = Optimizer::new(OptimizerMethod::Backprop, 10_000, rate)
            .with_log(Some(1_000))
            .with_batches(Some(2));

        assert_eq!(Ok(()), optim.train(&mut net, &inputs, &targets));

        let fin = net.mean_squared_error(&inputs, &targets).unwrap();

        println!("------------------");
        println!("Final cost: {fin}");

        let res = net.run_batch(&inputs).unwrap();
        for (inp, out) in inputs.to_vec().into_iter().zip(res.to_vec()) {
            println!("{:?} -> {}", inp.to_vec(), out[0])
        }

        assert!(fin < 0.01);
    }
}
