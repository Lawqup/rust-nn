use rust_nn::{
    matrix::Matrix2,
    neural::{
        activations::Activations,
        optimizer::{Optimizer, OptimizerMethod},
        NeuralNet,
    },
};

fn main() {
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

    let optim = Optimizer::new(OptimizerMethod::Backprop, 100_000, rate).with_log(Some(1));

    let _ = optim.train_gui(&mut net, &train_inputs, &train_targets);

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
