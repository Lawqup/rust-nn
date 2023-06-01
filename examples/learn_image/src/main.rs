use std::{
    fs::File,
    io::{BufReader, Read},
    sync::mpsc::Receiver,
};

use eframe::App;
use egui::{
    plot::{Line, Plot},
    CentralPanel,
};

use egui_extras::RetainedImage;
use image::{codecs::jpeg::JpegEncoder, GenericImageView, GrayImage, ImageBuffer, Luma, Pixel};
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

fn img_to_normalized_training_data(path: &str) -> (Matrix2<f64>, Matrix2<f64>) {
    let img = image::open(path).unwrap();

    let mut input = Vec::new();
    let mut targets = Vec::new();
    for (x, y, v) in img.pixels() {
        input.push(vec![
            x as f64 / img.width() as f64,
            y as f64 / img.height() as f64,
        ]);
        targets.push(vec![v.to_luma()[0] as f64 / 255.0]);
    }

    (
        Matrix2::from_vec(input).unwrap(),
        Matrix2::from_vec(targets).unwrap(),
    )
}

fn input_output_as_img(
    inputs: &Matrix2<f64>,
    outputs: &Matrix2<f64>,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    const WIDTH: u32 = 28;
    const HEIGHT: u32 = 28;

    let mut img = GrayImage::new(WIDTH, HEIGHT);

    for (x, y, v) in inputs
        .iter()
        .zip(outputs)
        .map(|(m1, m2)| (m1[0], m1[1], m2[0]))
    {
        img.put_pixel(
            (x * WIDTH as f64) as u32,
            (y * HEIGHT as f64) as u32,
            Luma([(v * 255.0) as u8]),
        );
    }

    img
}

lazy_static! {
    static ref INPUTS: Matrix2<f64> = img_to_normalized_training_data("assets/nine.jpg").0;
    static ref TARGETS: Matrix2<f64> = img_to_normalized_training_data("assets/nine.jpg").1;
}

struct ImgGui {
    nn_gui: NNGui,
    target: RetainedImage,
}

impl App for ImgGui {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        let (data, outputs) = self.nn_gui.get_data();

        if outputs.is_none() {
            return;
        }

        let outputs = outputs.unwrap();
        let data: Vec<_> = data.into_iter().map(|(i, j)| [i as f64, j]).collect();

        let mut buf = Vec::new();
        let img = input_output_as_img(&*INPUTS, &outputs);
        JpegEncoder::new(&mut buf).encode_image(&img).unwrap();
        let output_image = RetainedImage::from_image_bytes("Output Image", &buf).unwrap();

        CentralPanel::default().show(ctx, |ui| {
            ui.columns(2, |cols| {
                cols[0].vertical(|ui| {
                    let [i, j] = &data.last().unwrap_or(&[0.0, f64::INFINITY]);

                    ui.heading(format!("Iteration {} error: {j}", *i as usize));

                    Plot::new("Error").show(ui, |ui| ui.line(Line::new(data)));
                });

                cols[1].vertical_centered(|ui| {
                    self.target.show_scaled(ui, 10.0);

                    output_image.show_scaled(ui, 10.0);
                });
            });
        });
    }
}

impl Visualizer for ImgGui {
    fn new(cc: &eframe::CreationContext, rx: Receiver<IterationState>) -> Self {
        let target_file = File::open("assets/nine.jpg").unwrap();
        let mut reader = BufReader::new(target_file);
        let mut buf = Vec::new();

        reader.read_to_end(&mut buf).unwrap();
        let target = RetainedImage::from_image_bytes("Target Image", &buf).unwrap();

        Self {
            nn_gui: NNGui::new(cc, rx),
            target,
        }
    }
}

fn main() {
    let mut net = NeuralNet::new(2, 28, &Activations::Sigmoid);
    net.add_layer(1, &Activations::Sigmoid);

    let rate = 1.0;

    let optim = Optimizer::new(OptimizerMethod::Backprop, 10_000, rate).with_log(Some(1));

    let _ = optim.train_gui::<ImgGui>(&mut net, &INPUTS, &TARGETS);
    // let _ = optim.train(&mut net, &INPUTS, &TARGETS);

    let outputs = net.run_batch(&*INPUTS).unwrap();
    let _ = input_output_as_img(&*INPUTS, &outputs).save("output.jpg");
}