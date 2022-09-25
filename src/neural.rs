use std::ops::Deref;

use crate::matrix::{Dot, Matrix1, Matrix2, Transpose};
use rand::distributions::{Distribution, Uniform};

#[derive(Debug, PartialEq)]
pub enum NNError {
    /// Indicated input dimensions are incompatible
    InputErr,
    DataFormatErr,
}

pub struct DenseLayer<'a> {
    weights: Matrix2<f32>,
    biases: Matrix1<f32>,
    activation: Box<dyn Fn(f32) -> f32 + 'a>,
}

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
        let activation = Box::new(|x| x);

        Self {
            weights,
            biases,
            activation,
        }
    }

    /// Add an activation function of this layer
    pub fn with_activation<F: Fn(f32) -> f32 + 'a>(mut self, f: F) -> Self {
        self.activation = Box::new(f);
        self
    }

    /// Propogates a batch of inputs through the layer applying the activation function.
    pub fn run_batch(&self, input: &Matrix2<f32>) -> Result<Matrix2<f32>, NNError> {
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

    /// Returns the amount of neurons in the layer
    pub fn neuron_amount(&self) -> u32 {
        self.biases.size() as u32
    }

    /// Returns the amount of inputs this layer accepts
    pub fn input_amount(&self) -> u32 {
        self.weights.dim().1 as u32
    }
}

impl<'a> NeuralNet<'a> {
    /// Creates a 1-layer neural net with some number of neurons
    /// that can accept a certain number of inputs
    pub fn new(n_inputs: u32, n_neurons: u32, activation_function: fn(f32) -> f32) -> Self {
        Self {
            layers: vec![DenseLayer::new(n_inputs, n_neurons).with_activation(activation_function)],
        }
    }

    pub fn add_layer(&mut self, n_neurons: u32, activation_function: fn(f32) -> f32) {
        let n_inputs = self.layers.last().unwrap().neuron_amount();
        self.layers
            .push(DenseLayer::new(n_inputs, n_neurons).with_activation(activation_function));
    }

    /// Propogates a batch of inputs through layers
    pub fn run_batch(&self, input: &Matrix2<f32>) -> Result<Matrix2<f32>, NNError> {
        // NN has to be initialized with at least one layer
        let first_layer = self.layers.first().unwrap();

        if input.row_size() as u32 != first_layer.input_amount() {
            return Err(NNError::InputErr);
        }

        let mut next_input = first_layer.run_batch(input).unwrap();
        for layer in self.layers.iter().skip(1) {
            next_input = layer.run_batch(&next_input).unwrap();
        }
        Ok(next_input)
    }

    /// Normalizes an output such that each set of outputs in the batch add to 1.
    pub fn normalize(input: Matrix2<f32>) -> Matrix2<f32> {
        let mut normalized = Vec::new();
        for mut row in input {
            let sum: f32 = row.into_iter().sum();
            row.apply(|x| x / sum);
            normalized.push(row.to_vec());
        }

        Matrix2::from_vec(normalized).unwrap()
    }

    /// Calculates the mean categorical cross-entropy loss for a batch of inputs.
    /// Returns an InputErr if amount of predictions and class targets don't match.
    /// Returns a DataFormatErr if a class target does not fit in the prediction.
    pub fn loss_cross_entropy(
        prediction: &Matrix2<f32>,
        class_targets: &Matrix1<usize>,
    ) -> Result<f32, NNError> {
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

        Ok(loss.iter().sum::<f32>() / loss.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inside_unit_circle_data(axis_size: u32) -> (Matrix2<f32>, Matrix1<usize>) {
        let axis_size = axis_size as i64;
        let mut input = Vec::new();
        let mut class_targets = Vec::new();
        let func = |x, y| x * x + y * y;

        for x in -axis_size / 2..=axis_size / 2 {
            for y in -axis_size / 2..=axis_size / 2 {
                // fit data between -1 and 1;
                let (x, y) = (
                    (2 * x) as f32 / axis_size as f32,
                    (2 * y) as f32 / axis_size as f32,
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

    fn default_inputs() -> Matrix2<f32> {
        Matrix2::from_array([
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8],
            [-1.5, 2.7, 3.3, -0.8],
        ])
    }

    fn sigmoid(x: f32) -> f32 {
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
            let sum = row.into_iter().sum::<f32>();
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
}
