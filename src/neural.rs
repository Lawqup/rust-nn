use std::fmt::Debug;

use crate::matrix::{Dot, Matrix1, Matrix2, Transpose};
use rand::distributions::{Distribution, Uniform};

#[derive(Debug, PartialEq)]
pub enum NNError {
    /// Indicated input dimensions are incompatible
    InputErr,
    DataFormatErr,
}

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Identity,
    Sigmoid,
}

impl Activation {
    /// Returns activation function at x
    pub fn call(&self, x: f64) -> f64 {
        match self {
            Self::Identity => x,
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }

    /// Returns derivative of activation function at x
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Identity => 1.0,
            Self::Sigmoid => {
                let s_x = self.call(x);
                s_x * (1.0 - s_x)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct DenseLayer {
    weights: Matrix2<f64>,
    biases: Matrix1<f64>,
    activation: Activation,
}

#[derive(Debug, Clone)]
pub struct NeuralNet {
    layers: Vec<DenseLayer>,
}

impl DenseLayer {
    /// Initializes a layer given the number of inputs and neurons.
    /// Weights initialized as a random value in [-1.0, 1.0].
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
            activation: Activation::Identity,
        }
    }

    /// Add an activation function of this layer
    pub fn with_activation(mut self, activation: Activation) -> Self {
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
    pub fn new(n_inputs: u32, n_neurons: u32, activation: Activation) -> Self {
        Self {
            layers: vec![DenseLayer::new(n_inputs, n_neurons).with_activation(activation)],
        }
    }

    pub fn add_layer(&mut self, n_neurons: u32, activation: Activation) {
        let n_inputs = self.layers.last().unwrap().neuron_amount();
        self.layers
            .push(DenseLayer::new(n_inputs, n_neurons).with_activation(activation));
    }

    /// Propogates a batch of inputs through layers
    pub fn run_batch(&self, inputs: &Matrix2<f64>) -> Result<Matrix2<f64>, NNError> {
        // NN has to be initialized with at least one layer
        let first_layer = self.layers.first().unwrap();

        if inputs.row_size() as u32 != first_layer.input_amount() {
            return Err(NNError::InputErr);
        }

        let mut next_input = first_layer.run_batch(inputs).unwrap();
        for layer in self.layers.iter().skip(1) {
            next_input = layer.run_batch(&next_input).unwrap();
        }
        Ok(next_input)
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
        Ok(sum / targets.column_size() as f64)
    }

    pub fn train(
        &mut self,
        iterations: usize,
        eps: f64,
        rate: f64,
        inputs: &Matrix2<f64>,
        targets: &Matrix2<f64>,
    ) -> Result<(), NNError> {
        for _ in 0..iterations {
            let mut w_grads = Vec::new();
            let mut b_grads = Vec::new();

            let costs = &self.mean_squared_error(&inputs, &targets)?;
            for i in 0..self.layers.len() {
                w_grads.push(self.layers[i].weights.clone());
                b_grads.push(self.layers[i].biases.clone());

                for j in 0..self.layers[i].weights.column_size() {
                    for k in 0..self.layers[i].weights.row_size() {
                        let saved = self.layers[i].weights[j][k];
                        self.layers[i].weights[j][k] += eps;
                        let dw = (&self.mean_squared_error(&inputs, &targets)? - costs) / eps;

                        self.layers[i].weights[j][k] = saved;
                        w_grads[i][j][k] = -rate * dw;
                    }

                    let saved = self.layers[i].biases[j];
                    self.layers[i].biases[j] += eps;
                    let db = (&self.mean_squared_error(&inputs, &targets)? - costs) / eps;

                    self.layers[i].biases[j] = saved;
                    b_grads[i][j] = -rate * db;
                }
            }

            for i in 0..w_grads.len() {
                self.layers[i].weights = (&self.layers[i].weights + &w_grads[i]).unwrap();
                self.layers[i].biases = (&self.layers[i].biases + &b_grads[i]).unwrap();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
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
        let net = NeuralNet::new(4, 20, Activation::Sigmoid);

        let out = net.run_batch(&default_inputs()).unwrap();

        assert_eq!(out.dim(), (4, 20));
        assert!(out
            .into_iter()
            .all(|row| row.into_iter().all(|&x| x < 1.0 && x > 0.0)));
    }

    #[test]
    fn train_or() {
        // Train a single neuron to compute OR
        let mut net = NeuralNet::new(2, 1, Activation::Sigmoid);

        let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
        let targets = Matrix2::from_array([[0], [1], [1], [1]]).into();

        let eps = 10e-4;
        let rate = 10e-2;

        assert_eq!(Ok(()), net.train(10000, eps, rate, &inputs, &targets));

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
        let mut net = NeuralNet::new(2, 2, Activation::Sigmoid);
        net.add_layer(1, Activation::Sigmoid);

        let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
        let targets = Matrix2::from_array([[0], [1], [1], [0]]).into();

        let eps = 10e-3;
        let rate = 10e-1;

        assert_eq!(Ok(()), net.train(200 * 100, eps, rate, &inputs, &targets));

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
        let mut net = NeuralNet::new(2, 2, Activation::Sigmoid);
        net.add_layer(2, Activation::Sigmoid);

        let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
        let targets = Matrix2::from_array([[0, 0], [1, 0], [1, 0], [0, 1]]).into();

        let eps = 10e-2;
        let rate = 10e-2;

        assert_eq!(Ok(()), net.train(100 * 1000, eps, rate, &inputs, &targets));

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
        let mut net = NeuralNet::new(4, 4, Activation::Sigmoid);
        net.add_layer(5, Activation::Sigmoid);
        net.add_layer(3, Activation::Sigmoid);

        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for i in 0..4 {
            for j in 0..4 {
                inputs.push(vec![i / 2, i % 2, j / 2, j % 2]);
                targets.push(vec![(i + j) / 4, ((i + j) / 2) % 2, (i + j) % 2])
            }
        }

        println!("{:?}\n{:?}", inputs, targets);

        let inputs = Matrix2::from_vec(inputs).unwrap().into();
        let targets = Matrix2::from_vec(targets).unwrap().into();

        let eps = 10e-1;
        let rate = 10e-1;

        assert_eq!(Ok(()), net.train(100 * 100, eps, rate, &inputs, &targets));

        let fin = net.mean_squared_error(&inputs, &targets).unwrap();

        println!("------------------");
        println!("Final cost: {fin}");

        let res = net.run_batch(&inputs).unwrap();
        for (inp, out) in inputs.iter().zip(res) {
            println!("{:?} -> {:?}", inp.as_vec(), out.as_vec())
        }
    }
}
