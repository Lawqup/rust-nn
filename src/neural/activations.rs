pub trait Activation {
    /// Returns activation function at x
    fn call(&self, x: f64) -> f64;
    /// Returns derivative of activation function with respect to the function at x.
    /// For example, if our activation is sigmoid, then we would express the
    /// derivative as `a_x * (1-a_x)` instead of `sigmoid(a_x)(1-sigmoid(a_x))`.
    fn derivative(&self, a_x: f64) -> f64;
}

#[derive(Debug, Clone, Copy)]
pub enum Activations {
    Identity,
    Sigmoid,
    Arctan,
    ReLU,
}

impl Activation for Activations {
    fn call(&self, x: f64) -> f64 {
        use Activations::*;
        match self {
            Identity => x,
            Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Arctan => x.atan(),
            ReLU => x.max(0.0),
        }
    }

    fn derivative(&self, a_x: f64) -> f64 {
        use Activations::*;
        match self {
            Identity => 1.0,
            Arctan => 1.0 / (1.0 + a_x.tan() * a_x.tan()),
            Sigmoid => a_x * (1.0 - a_x),
            ReLU => {
                if a_x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}
