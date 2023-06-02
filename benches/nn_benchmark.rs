use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_nn::{
    matrix::Matrix2,
    neural::{
        activations::Activations,
        optimizer::{Optimizer, OptimizerMethod},
        NeuralNet,
    },
};

fn train_tiny(iterations: usize) {
    let mut net = NeuralNet::new(2, 2, &Activations::Arctan);
    net.add_layer(1, &Activations::Sigmoid);

    let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
    let targets = Matrix2::from_array([[0], [1], [1], [0]]).into();

    let rate = 10e-0;

    let optim = Optimizer::new(OptimizerMethod::Backprop, iterations, rate);

    assert_eq!(Ok(()), optim.train(&mut net, &inputs, &targets));
}

fn train_small(iterations: usize) {
    let mut net = NeuralNet::new(2, 10, &Activations::Sigmoid);
    net.add_layer(10, &Activations::ReLU);
    net.add_layer(2, &Activations::Sigmoid);

    let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
    let targets = Matrix2::from_array([[0, 0], [1, 0], [1, 0], [0, 1]]).into();

    let rate = 10e-3;

    let optim = Optimizer::new(OptimizerMethod::Backprop, iterations, rate);
    assert_eq!(Ok(()), optim.train(&mut net, &inputs, &targets));
}

fn train_medium(iterations: usize) {
    let mut net = NeuralNet::new(2, 20, &Activations::Sigmoid);
    net.add_layer(20, &Activations::Arctan);
    net.add_layer(20, &Activations::Arctan);
    net.add_layer(2, &Activations::Sigmoid);

    let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
    let targets = Matrix2::from_array([[0, 0], [1, 0], [1, 0], [0, 1]]).into();

    let rate = 10e-3;

    let optim = Optimizer::new(OptimizerMethod::Backprop, iterations, rate);
    assert_eq!(Ok(()), optim.train(&mut net, &inputs, &targets));
}

fn forward(net: &NeuralNet, inputs: &Matrix2<f64>) {
    for i in 0..inputs.rows() {
        assert!(net.forward(&inputs.clone_row(i)).is_ok());
    }
}

fn bench_forward(c: &mut Criterion) {
    let mut small = NeuralNet::new(2, 10, &Activations::Sigmoid);
    small.add_layer(10, &Activations::Arctan);
    small.add_layer(2, &Activations::Arctan);

    let mut medium = NeuralNet::new(2, 20, &Activations::Sigmoid);
    medium.add_layer(20, &Activations::Arctan);
    medium.add_layer(20, &Activations::Arctan);
    medium.add_layer(2, &Activations::Sigmoid);

    let input_small = Matrix2::new(10, 2);
    let input_medium = Matrix2::new(1_000, 2);

    c.bench_function("forward small 10 inputs", |b| {
        b.iter(|| forward(black_box(&small), black_box(&input_small)))
    });
    c.bench_function("forward small 1,000 inputs", |b| {
        b.iter(|| forward(black_box(&small), black_box(&input_medium)))
    });

    c.bench_function("forward medium 10 inputs", |b| {
        b.iter(|| forward(black_box(&medium), black_box(&input_small)))
    });
    c.bench_function("forward medium 1,000 inputs", |b| {
        b.iter(|| forward(black_box(&medium), black_box(&input_medium)))
    });
}

fn bench_tiny(c: &mut Criterion) {
    c.bench_function("tiny 10 iterations", |b| {
        b.iter(|| train_tiny(black_box(10)))
    });
    c.bench_function("tiny 10,000 iterations", |b| {
        b.iter(|| train_tiny(black_box(10_000)))
    });
}

fn bench_small(c: &mut Criterion) {
    c.bench_function("small 10 iterations", |b| {
        b.iter(|| train_small(black_box(10)))
    });
    c.bench_function("small 10,000 iterations", |b| {
        b.iter(|| train_small(black_box(10_000)))
    });
}

fn bench_medium(c: &mut Criterion) {
    c.bench_function("medium 10 iterations", |b| {
        b.iter(|| train_medium(black_box(10)))
    });
    c.bench_function("medium 10,000 iterations", |b| {
        b.iter(|| train_medium(black_box(10_000)))
    });
}

criterion_group!(
    benches,
    bench_forward,
    bench_tiny,
    bench_small,
    bench_medium
);
criterion_main!(benches);
