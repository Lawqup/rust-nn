use crate::matrix::Matrix2;

use super::NeuralNet;

pub struct Grad(
    pub Vec<Vec<Matrix2<f64>>>,
    pub Vec<Vec<Matrix2<f64>>>,
    pub Vec<Matrix2<f64>>,
);

impl Grad {
    pub fn empty(net: &NeuralNet, n_inputs: usize) -> Self {
        let mut w_grads = Vec::with_capacity(n_inputs);
        let mut b_grads = Vec::with_capacity(n_inputs);
        let mut d_acts = Vec::with_capacity(net.layers.len() + 1);

        for _ in 0..n_inputs {
            let mut w_grad = Vec::with_capacity(net.layers.len());
            let mut b_grad = Vec::with_capacity(net.layers.len());

            for layer in &net.layers {
                w_grad.push(Matrix2::new(layer.weights.rows(), layer.weights.cols()));
                b_grad.push(Matrix2::new(1, layer.biases.cols()));
            }

            w_grads.push(w_grad);
            b_grads.push(b_grad);
        }

        for l in &net.layers {
            d_acts.push(Matrix2::new(1, l.weights.cols()));
        }
        d_acts.push(Matrix2::new(0, 0));

        Grad(w_grads, b_grads, d_acts)
    }
}
