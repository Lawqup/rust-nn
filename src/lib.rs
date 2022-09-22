pub mod matrix;

use matrix::{Matrix1, Matrix2};

struct NeuralNet {
    weights: Matrix2<f32>,
    biases: Matrix1<f32>,
}

impl NeuralNet {
    fn new() -> Self {
        Self {
            weights: Matrix2::from_array([[1.0; 4]; 3]),
            biases: Matrix1::from_array([1.0; 3]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let net = NeuralNet::new();
    }
}
