pub mod activations;
pub mod optimizer;

use std::fmt::Debug;

use crate::matrix::{Dot, Matrix1, Matrix2, Transpose};
use rand::distributions::{Distribution, Uniform};

use self::activations::{Activation, Activations};

#[derive(Debug, PartialEq)]
pub enum NNError {
    /// Indicated input dimensions are incompatible
    InputErr,
    DataFormatErr,
}

pub struct DenseLayer<'a> {
    weights: Matrix2<f64>,
    biases: Matrix1<f64>,
    activation: &'a dyn Activation,
}

pub struct NeuralNet<'a> {
    layers: Vec<DenseLayer<'a>>,
}

impl<'a> DenseLayer<'a> {
    /// Initializes a layer given the number of inputs and neurons.
    /// Weights initialized as a random value in [-1.0, 1.0]
    /// Biases initialized as 0.
    /// Activation function is initally the identity (f(x) = x)
    pub fn new(n_inputs: u32, n_neurons: u32) -> Self {
        let mut rng = rand::thread_rng();
        let die = Uniform::from(-1.0..=1.0);

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

        let biases = Matrix1::from_vec((0..n_neurons).map(|_| 0.0).collect());

        Self {
            weights,
            biases,
            activation: &Activations::Identity,
        }
    }

    /// Add an activation function of this layer
    pub fn with_activation(mut self, activation: &'a impl Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Propogates a batch of inputs through the layer applying the activation function.
    pub fn run_batch(&self, input: &Matrix2<f64>) -> Result<Matrix2<f64>, NNError> {
        match input
            .dot(&self.weights.transpose())
            .and_then(|m| &m + &self.biases)
        {
            Ok(mut out) => {
                out.apply(|x| self.activation.call(x));
                Ok(out)
            }
            Err(_) => Err(NNError::InputErr),
        }
    }

    /// Propogates an input through the layer applying the activation function.
    pub fn forward(&self, input: &Matrix1<f64>) -> Result<Matrix1<f64>, NNError> {
        match self.weights.dot(input).and_then(|m| &m + &self.biases) {
            Ok(mut out) => {
                out.apply(|x| self.activation.call(x));
                Ok(out)
            }
            Err(_) => Err(NNError::InputErr),
        }
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

impl<'a> NeuralNet<'a> {
    /// Creates a 1-layer neural net with some number of neurons
    /// that can accept a certain number of inputs
    pub fn new(n_inputs: u32, n_neurons: u32, activation: &'a impl Activation) -> Self {
        Self {
            layers: vec![DenseLayer::new(n_inputs, n_neurons).with_activation(activation)],
        }
    }

    pub fn add_layer(&mut self, n_neurons: u32, activation: &'a impl Activation) {
        let n_inputs = self.layers.last().unwrap().neuron_amount();
        self.layers
            .push(DenseLayer::new(n_inputs, n_neurons).with_activation(activation));
    }

    pub fn forward(&self, input: &Matrix1<f64>) -> Result<Vec<Matrix1<f64>>, NNError> {
        // NN has to be initialized with at least one layer
        let first_layer = self.layers.first().unwrap();

        if input.size() as u32 != first_layer.input_amount() {
            return Err(NNError::InputErr);
        }

        let mut outputs = vec![input.clone()];
        for layer in self.layers.iter() {
            outputs.push(layer.forward(&outputs[outputs.len() - 1]).unwrap());
        }
        Ok(outputs)
    }

    /// Propogates a batch of inputs through layers
    pub fn run_batch(&self, inputs: &Matrix2<f64>) -> Result<Matrix2<f64>, NNError> {
        // NN has to be initialized with at least one layer
        let first_layer = self.layers.first().unwrap();

        if inputs.cols() as u32 != first_layer.input_amount() {
            return Err(NNError::InputErr);
        }

        let mut prev_output = first_layer.run_batch(inputs).unwrap();
        for layer in self.layers.iter().skip(1) {
            prev_output = layer.run_batch(&prev_output).unwrap();
        }
        Ok(prev_output)
    }

    /// Normalizes an output such that each set of outputs in the batch add to 1.
    pub fn normalize(input: Matrix2<f64>) -> Matrix2<f64> {
        let mut normalized = Vec::new();
        for mut row in input {
            let sum: f64 = row.into_iter().sum();
            row.apply(|x| x / sum);
            normalized.push(row.to_vec());
        }

        Matrix2::from_vec(normalized).unwrap()
    }

    /// Mean-squared error
    pub fn mean_squared_error(
        &self,
        inputs: &Matrix2<f64>,
        targets: &Matrix2<f64>,
    ) -> Result<f64, NNError> {
        // Prediction rows must match class target size
        let outputs = self.run_batch(inputs)?;
        if outputs.dim() != targets.dim() {
            return Err(NNError::InputErr);
        }

        let sum: f64 = outputs
            .iter()
            .zip(targets.iter())
            .map(|(y, y_hat)| {
                let mut diff = (y - y_hat).unwrap();
                diff.apply(|x| x * x);
                diff.iter().sum::<f64>() / diff.size() as f64
            })
            .sum();
        Ok(sum / targets.rows() as f64)
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
        let layer1 = DenseLayer::new(4, 2);
        let layer2 = DenseLayer::new(2, 1);

        let layer1_out = layer1.run_batch(&default_inputs()).unwrap();

        let layer2_out = layer2.run_batch(&layer1_out).unwrap();

        assert_eq!(layer2_out.dim(), (4, 1));
        assert_eq!(layer2_out[2].as_vec(), layer2_out[3].as_vec());
    }

    #[test]
    fn activation_function() {
        let net = NeuralNet::new(4, 20, &Activations::Sigmoid);

        let out = net.run_batch(&default_inputs()).unwrap();

        assert_eq!(out.dim(), (4, 20));
        assert!(out
            .into_iter()
            .all(|row| row.into_iter().all(|&x| x < 1.0 && x > 0.0)));
    }

    #[test]
    fn train_or() {
        // Train a single neuron to compute OR
        let mut net = NeuralNet::new(2, 1, &Activations::Sigmoid);

        let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
        let targets = Matrix2::from_array([[0], [1], [1], [1]]).into();

        let rate = 10e-0;

        let optim = Optimizer::new(OptimizerMethod::Backprop, 1_000, rate);
        assert_eq!(Ok(()), optim.train(&mut net, &inputs, &targets));

        let fin = net.mean_squared_error(&inputs, &targets).unwrap();

        println!("------------------");
        println!("Final cost: {fin}");

        let res = net.run_batch(&inputs).unwrap();
        for (inp, out) in inputs.iter().zip(res) {
            println!("{:?} -> {}", inp.as_vec(), out[0])
        }

        assert!(fin < 0.01);
    }

    #[test]
    fn train_xor() {
        let mut net = NeuralNet::new(2, 2, &Activations::Arctan);
        net.add_layer(1, &Activations::Sigmoid);

        let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
        let targets = Matrix2::from_array([[0], [1], [1], [0]]).into();

        let rate = 10e-0;

        let optim = Optimizer::new(OptimizerMethod::Backprop, 10_000, rate).with_log(Some(1_000));
        assert_eq!(Ok(()), optim.train(&mut net, &inputs, &targets));

        let fin = net.mean_squared_error(&inputs, &targets).unwrap();

        println!("------------------");
        println!("Final cost: {fin}");

        let res = net.run_batch(&inputs).unwrap();
        for (inp, out) in inputs.iter().zip(res) {
            println!("{:?} -> {}", inp.as_vec(), out[0])
        }

        assert!(fin < 0.01);
    }

    #[test]
    fn train_multi_in_multi_out() {
        let mut net = NeuralNet::new(2, 2, &Activations::Sigmoid);
        net.add_layer(2, &Activations::Sigmoid);

        let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
        let targets = Matrix2::from_array([[0, 0], [1, 0], [1, 0], [0, 1]]).into();

        let rate = 10e-3;

        let optim = Optimizer::new(OptimizerMethod::Backprop, 100_000, rate).with_log(Some(5_000));
        assert_eq!(Ok(()), optim.train(&mut net, &inputs, &targets));

        let fin = net.mean_squared_error(&inputs, &targets).unwrap();

        println!("------------------");
        println!("Final cost: {fin}");

        let res = net.run_batch(&inputs).unwrap();
        for (inp, out) in inputs.iter().zip(res) {
            println!("{:?} -> {:?}", inp.as_vec(), out.as_vec())
        }

        assert!(fin < 0.001);
    }

    #[test]
    fn train_adder() {
        const BITS: u32 = 3; // Number of bits per number to add
        let mut net = NeuralNet::new(2 * BITS, 4, &Activations::Sigmoid);
        net.add_layer(2 * BITS + 1, &Activations::Sigmoid);
        net.add_layer(BITS + 1, &Activations::Sigmoid);

        let mut train_inputs = Vec::new();
        let mut train_targets = Vec::new();

        fn to_bitvec(x: i32, size: u32) -> Vec<i32> {
            (0..size).map(|i| x >> i & 1).collect()
        }
        for i in 0..1 << BITS {
            for j in 0..1 << BITS {
                train_inputs.push([to_bitvec(i, BITS), to_bitvec(j, BITS)].concat());
                train_targets.push(to_bitvec(i + j, BITS + 1))
            }
        }

        let train_inputs = Matrix2::from_vec(train_inputs).unwrap().into();
        let train_targets = Matrix2::from_vec(train_targets).unwrap().into();

        let rate = 1.0;

        let optim = Optimizer::new(OptimizerMethod::Backprop, 10_000, rate).with_log(Some(5_000));
        assert_eq!(Ok(()), optim.train(&mut net, &train_inputs, &train_targets));

        println!("------------------");
        println!(
            "Final training cost: {}",
            net.mean_squared_error(&train_inputs, &train_targets)
                .unwrap()
        );

        let res = net.run_batch(&train_inputs).unwrap();

        fn from_bits(bv: Vec<f64>) -> usize {
            bv.into_iter()
                .enumerate()
                .fold(0, |acc, (idx, v)| acc + ((v.round() as usize) << idx))
        }
        let mut correct = 0.0;
        let inps = train_inputs.rows();
        for (idx, mut i) in train_inputs.into_iter().map(|i| i.to_vec()).enumerate() {
            let y = from_bits(res[idx].as_vec().clone());
            let j = from_bits(i.split_off(BITS as usize));
            let i = from_bits(i);
            if i + j != y {
                println!("{i}+{j} != {y}",);
            } else {
                correct += 1.0;
            }
        }

        println!("Accuracy = {}", correct / inps as f64);
    }
}
