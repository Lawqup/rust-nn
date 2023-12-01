# Rust NN: an educational Neural Network framework in Rust

<img src="./assets/learn_image.gif" width="50%" height="50%">

Features include:

- Ergonomic APIs for building and training your networks.
- Predefined optimizer methods and activation functions, as well as the ability to add your own via implementing a trait.
- Fully customizable GUI via [egui](https://github.com/emilk/egui) for seeing your network during training, including some prebuilt GUIs.

## Examples

For more detailed examples, look at the examples directory.

- [Examples directory](./examples/). Featured examples:
  - [adder_gui](./examples/learn_image) -- a simple example of creating a custom training GUI to train a neural network to add numbers
  - [learn_image](./examples/learn_image) -- a more involved example for training a neural network to upscale and interpolate images

**Note**: Running in release makes training *drastically* faster, i.e. do `cargo run --release`.

### Minimal example

Here is how you'd create a model that learns to behave like an XOR gate.

```Rust
let mut net = NeuralNet::new(2, 2, Activations::Arctan);
net.add_layer(1, Activations::Sigmoid);

let inputs = Matrix2::from_array([[0, 0], [0, 1], [1, 0], [1, 1]]).into();
let targets = Matrix2::from_array([[0], [1], [1], [0]]).into();

let learning_rate = 10e-0;
let epochs = 10_000;

// Initialize the optimizer to log the error at each epoch
let optim = Optimizer::new(OptimizerMethod::Backprop, learning_rate, epochs).with_log(Some(1));
        
optim.train(&mut net, &inputs, &targets);

let fin = net.mean_squared_error(&inputs, &targets).unwrap();

println!("------------------");
let res = net.run_batch(&inputs).unwrap();
for (inp, out) in inputs.to_vec().into_iter().zip(res.to_vec()) {
    println!("{:?} -> {}", inp.to_vec(), out[0])
}
```

To make the optimizer display graph the error at each epoch, you can do this:

```Rust
optim.train_gui<NNGui>(&mut net, &inputs, &targets);
```

Here, `NNGui` is a predefined struct implementing the `Visualizer` trait that allows the optimizer to spawn and asynchronously update a GUI (meaning performance shouldn't be affected).
