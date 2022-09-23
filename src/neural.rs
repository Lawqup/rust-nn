use crate::matrix::{Dot, Matrix1, Matrix2, Transpose};
use rand::distributions::{Distribution, Uniform};

#[derive(Debug)]
pub enum NNError {
    /// Indicated input dimensions are incompatible
    InputErr,
}

pub struct DenseLayer {
    weights: Matrix2<f32>,
    biases: Matrix1<f32>,
    activation: fn(f32) -> f32,
}

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

        // No activation function
        let activation = |x| x;

        Self {
            weights,
            biases,
            activation,
        }
    }

    /// Add an activation function of this layer
    pub fn with_activation(mut self, f: fn(f32) -> f32) -> Self {
        self.activation = f;
        self
    }

    /// Propogates a batch of inputs through the layer applying the activation function.
    pub fn run_batch(&self, input: &Matrix2<f32>) -> Result<Matrix2<f32>, NNError> {
        match input
            .dot(&self.weights.transpose())
            .and_then(|m| &m + &self.biases)
        {
            Ok(mut out) => {
                out.apply(self.activation);
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

impl NeuralNet {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
