use std::sync::mpsc::Receiver;

use eframe::App;
use egui::{
    plot::{Line, Plot},
    CentralPanel,
};

use lazy_static::lazy_static;
use rust_nn::{
    matrix::Matrix2,
    neural::{
        activations::Activations,
        optimizer::{Optimizer, OptimizerMethod},
        NeuralNet,
    },
    viz::{IterationState, NNGui, Visualizer},
};

const BITS: u32 = 3; // Number of bits per number to add

fn to_bitvec(x: i32, size: u32) -> Vec<i32> {
    (0..size).map(|i| x >> i & 1).collect()
}

lazy_static! {
    static ref INPUTS: Matrix2<f64> = {
        let mut inputs = Vec::new();

        for i in 0..1 << BITS {
            for j in 0..1 << BITS {
                inputs.push([to_bitvec(i, BITS), to_bitvec(j, BITS)].concat());
            }
        }

        Matrix2::from_vec(inputs).unwrap().into()
    };
    static ref TARGETS: Matrix2<f64> = {
        let mut targets = Vec::new();

        for i in 0..1 << BITS {
            for j in 0..1 << BITS {
                targets.push(to_bitvec(i + j, BITS + 1))
            }
        }

        Matrix2::from_vec(targets).unwrap().into()
    };
}

struct AccGui(NNGui);

impl App for AccGui {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        let data = self.0.get_data();

        let res = data
            .last()
            .map(|(_, _, k)| k.clone())
            .unwrap_or(Matrix2::new(0, 0));
        let data: Vec<_> = data.into_iter().map(|(i, j, _)| [i as f64, j]).collect();

        CentralPanel::default().show(ctx, |ui| {
            ui.columns(2, |cols| {
                cols[0].vertical(|ui| {
                    let [i, j] = &data.last().unwrap_or(&[0.0, f64::INFINITY]);
                    ui.heading(format!("Iteration {} error: {j}", *i as usize));
                    Plot::new("Error").show(ui, |ui| ui.line(Line::new(data)));
                });

                let mut incorrect = Vec::new();

                fn from_bits(bv: &Vec<f64>) -> usize {
                    bv.into_iter()
                        .enumerate()
                        .fold(0, |acc, (idx, v)| acc + ((v.round() as usize) << idx))
                }
                let mut correct = 0.0;
                let inps = INPUTS.rows();
                for (idx, mut i) in INPUTS.iter().map(|i| i.as_vec().clone()).enumerate() {
                    let y = from_bits(res[idx].as_vec());
                    let j = from_bits(&i.split_off(3 as usize));
                    let i = from_bits(&i);
                    if i + j != y {
                        incorrect.push(format!("{i} + {j} != {y}"))
                    } else {
                        correct += 1.0;
                    }
                }
                cols[1].vertical(|ui| {
                    ui.heading(format!("Accuracy = {}", correct / inps as f64));
                    for inc in incorrect {
                        ui.small(inc);
                    }
                });
            });
        });
    }
}

impl Visualizer for AccGui {
    fn new(cc: &eframe::CreationContext, rx: Receiver<IterationState>) -> Self {
        Self(NNGui::new(cc, rx))
    }
}

fn main() {
    let mut net = NeuralNet::new(2 * BITS, 4, &Activations::Sigmoid);
    net.add_layer(3 * BITS + 1, &Activations::ReLU);
    net.add_layer(BITS + 1, &Activations::Sigmoid);

    let rate = 1.0;

    let optim = Optimizer::new(OptimizerMethod::Backprop, 10_000, rate).with_log(Some(1));

    let _ = optim.train_gui::<AccGui>(&mut net, &INPUTS, &TARGETS);
}
