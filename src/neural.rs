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
}

pub struct NeuralNet {
    layers: Vec<DenseLayer>,
}

impl DenseLayer {
    pub fn new(n_inputs: u32, n_neurons: u32) -> Self {
        let mut rng = rand::thread_rng();
        let die = Uniform::from(0.0..=1.0);

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
        Self { weights, biases }
    }

    pub fn run_batch(&self, input: &Matrix2<f32>) -> Result<Matrix2<f32>, NNError> {
        match input
            .dot(&self.weights.transpose())
            .and_then(|m| &m + &self.biases)
        {
            Ok(out) => Ok(out),
            Err(_) => Err(NNError::InputErr),
        }
    }
}

impl NeuralNet {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
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

    #[test]
    fn one_layer() {
        let layer = DenseLayer::new(4, 3);

        let output = layer.run_batch(&default_inputs()).unwrap();

        assert_eq!(output.dim(), (4, 3));
        assert_eq!(output[2].as_vec(), output[3].as_vec());
    }

    #[test]
    fn two_layer() {
        let layer1 = DenseLayer::new(4, 2);
        let layer2 = DenseLayer::new(2, 1);

        let layer1_out = layer1.run_batch(&default_inputs()).unwrap();

        let layer2_out = layer2.run_batch(&layer1_out).unwrap();

        assert_eq!(layer2_out.dim(), (4, 1));
        assert_eq!(layer2_out[2].as_vec(), layer2_out[3].as_vec());
    }
}
