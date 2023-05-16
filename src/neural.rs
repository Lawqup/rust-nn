use std::{fmt::Debug, ops::Deref, rc::Rc};

use crate::matrix::{Dot, Matrix1, Matrix2, Transpose};
use rand::distributions::{Distribution, Uniform};

#[derive(Debug, PartialEq)]
pub enum NNError {
    /// Indicated input dimensions are incompatible
    InputErr,
    DataFormatErr,
}

#[derive(Clone)]
pub struct DenseLayer<'a> {
    weights: Matrix2<f64>,
    biases: Matrix1<f64>,
    activation: Rc<dyn Fn(f64) -> f64 + 'a>,
}

#[derive(Clone)]
pub struct NeuralNet<'a> {
    layers: Vec<DenseLayer<'a>>,
}

impl<'a> DenseLayer<'a> {
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

        // No activation function
        let activation = Rc::new(|x| x);

        Self {
            weights,
            biases,
            activation,
        }
    }

    /// Add an activation function of this layer
    pub fn with_activation<F: Fn(f64) -> f64 + 'a>(mut self, f: F) -> Self {
        self.activation = Rc::new(f);
        self
    }

    /// Propogates a batch of inputs through the layer applying the activation function.
    pub fn run_batch(&self, input: &Matrix2<f64>) -> Result<Matrix2<f64>, NNError> {
        match input
            .dot(&self.weights.transpose())
            .and_then(|m| &m + &self.biases)
        {
            Ok(mut out) => {
                out.apply(self.activation.deref());
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
    pub fn new(n_inputs: u32, n_neurons: u32, activation_function: fn(f64) -> f64) -> Self {
        Self {
            layers: vec![DenseLayer::new(n_inputs, n_neurons).with_activation(activation_function)],
        }
    }

    pub fn add_layer(&mut self, n_neurons: u32, activation_function: fn(f64) -> f64) {
        let n_inputs = self.layers.last().unwrap().neuron_amount();
        self.layers
            .push(DenseLayer::new(n_inputs, n_neurons).with_activation(activation_function));
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

    /// Calculates the mean categorical cross-entropy loss for a batch of inputs.
    /// Returns an InputErr if amount of predictions and class targets don't match.
    /// Returns a DataFormatErr if a class target does not fit in the prediction.
    pub fn loss_cross_entropy(
        prediction: &Matrix2<f64>,
        class_targets: &Matrix1<usize>,
    ) -> Result<f64, NNError> {
        // Prediction rows must match class target size
        if prediction.column_size() != class_targets.size() {
            return Err(NNError::InputErr);
        }

        let mut loss = Vec::new();
        for (pred, &target) in prediction.iter().zip(class_targets) {
            // Target doesnt exist in prediction
            if target >= pred.size() {
                return Err(NNError::DataFormatErr);
            }

            loss.push(-pred[target].clamp(1e-7, 1.0).ln());
        }

        Ok(loss.iter().sum::<f64>() / loss.len() as f64)
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
    ) {
        for _ in 0..iterations {
            let mut w_grads = Vec::new();
            let mut b_grads = Vec::new();

            let costs = &self.mean_squared_error(&inputs, &targets).unwrap();
            for i in 0..self.layers.len() {
                w_grads.push(self.layers[i].weights.clone());
                b_grads.push(self.layers[i].biases.clone());

                for j in 0..self.layers[i].weights.column_size() {
                    for k in 0..self.layers[i].weights.row_size() {
                        let saved = self.layers[i].weights[j][k];
                        self.layers[i].weights[j][k] += eps;
                        let dw =
                            (&self.mean_squared_error(&inputs, &targets).unwrap() - costs) / eps;
                        self.layers[i].weights[j][k] = saved;
                        w_grads[i][j][k] = -rate * dw;
                    }

                    let saved = self.layers[i].biases[j];
                    self.layers[i].biases[j] += eps;
                    let db = (&self.mean_squared_error(&inputs, &targets).unwrap() - costs) / eps;
                    self.layers[i].biases[j] = saved;
                    b_grads[i][j] = -rate * db;
                }
            }

            for i in 0..w_grads.len() {
                self.layers[i].weights = (&self.layers[i].weights + &w_grads[i]).unwrap();
                self.layers[i].biases = (&self.layers[i].biases + &b_grads[i]).unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inside_unit_circle_data(axis_size: u32) -> (Matrix2<f64>, Matrix1<usize>) {
        let axis_size = axis_size as i64;
        let mut input = Vec::new();
        let mut class_targets = Vec::new();
        let func = |x, y| x * x + y * y;

        for x in -axis_size / 2..=axis_size / 2 {
            for y in -axis_size / 2..=axis_size / 2 {
                // fit data between -1 and 1;
                let (x, y) = (
                    (2 * x) as f64 / axis_size as f64,
                    (2 * y) as f64 / axis_size as f64,
                );
                input.push(vec![x, y]);

                // 0th class => in circle
                // 1st class => outside of circle
                class_targets.push((func(x, y) > 1.0) as usize);
            }
        }

        (
            Matrix2::from_vec(input).unwrap(),
            Matrix1::from_vec(class_targets),
        )
    }

    fn default_inputs() -> Matrix2<f64> {
        Matrix2::from_array([
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8],
            [-1.5, 2.7, 3.3, -0.8],
        ])
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
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
        let net = NeuralNet::new(4, 20, sigmoid);

        let out = net.run_batch(&default_inputs()).unwrap();

        assert_eq!(out.dim(), (4, 20));
        assert!(out
            .into_iter()
            .all(|row| row.into_iter().all(|&x| x < 1.0 && x > 0.0)));
    }

    #[test]
    fn softmax() {
        let mut net = NeuralNet::new(2, 10, |x| x.max(0.0));
        net.add_layer(2, |x| x.exp());

        let out = net.run_batch(&inside_unit_circle_data(1000).0).unwrap();

        let normalized = NeuralNet::normalize(out);

        let delta = 0.00001;

        assert!(normalized.iter().all(|row| {
            let sum = row.into_iter().sum::<f64>();
            sum < 1.0 + delta && sum > 1.0 - delta
        }));
    }

    #[test]
    fn categorical_cross_entropy() {
        let mut net = NeuralNet::new(2, 10, |x| x.max(0.0));
        net.add_layer(2, |x| x.exp());

        let (input, targets) = &inside_unit_circle_data(30);

        let out = net.run_batch(input).unwrap();

        let normalized = &NeuralNet::normalize(out);

        let loss = NeuralNet::loss_cross_entropy(normalized, targets).unwrap();

        assert!(loss > 0.0);

        let normalized = &Matrix2::from_array([[0.7, 0.3]]);
        let targets = &Matrix1::from_array([2]);

        let loss = NeuralNet::loss_cross_entropy(normalized, targets);

        assert_eq!(loss, Err(NNError::DataFormatErr));
    }

    #[test]
    fn train_or() {
        // Train a single neuron to compute OR
        let mut net = NeuralNet::new(2, 1, sigmoid);

        let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
        let targets = Matrix2::from_array([[0], [1], [1], [1]]).into();

        let eps = 10e-4;
        let rate = 10e-2;

        net.train(10000, eps, rate, &inputs, &targets);

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
        let mut net = NeuralNet::new(2, 2, sigmoid);
        net.add_layer(1, sigmoid);

        let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
        let targets = Matrix2::from_array([[0], [1], [1], [0]]).into();

        let eps = 10e-2;
        let rate = 10e-2;

        net.train(100 * 1000, eps, rate, &inputs, &targets);

        let fin = net.mean_squared_error(&inputs, &targets).unwrap();

        println!("------------------");
        println!("Final cost: {fin}");

        let res = net.run_batch(&inputs).unwrap();
        for (inp, out) in inputs.iter().zip(res) {
            println!("{:?} -> {}", inp.as_vec(), out[0])
        }

        assert!(fin < 0.001);
    }
}
