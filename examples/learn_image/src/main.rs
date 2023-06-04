use std::{
    fs::File,
    io::{BufReader, Read},
    sync::mpsc::Receiver,
};

use eframe::App;
use egui::{
    plot::{Line, Plot},
    CentralPanel, Slider,
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
    viz::{EpochState, NNGui, Visualizer},
};

fn get_normalized_training_data(path: &str, third_dim: f64) -> (Matrix2<f64>, Matrix2<f64>) {
    let img = image::open(path).unwrap();

    let mut input = Vec::new();
    let mut targets = Vec::new();
    for (x, y, v) in img.pixels() {
        input.push(vec![
            x as f64 / img.width() as f64,
            y as f64 / img.height() as f64,
            third_dim,
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
    width: u32,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let mut img = GrayImage::new(width, width);

    for (x, y, v) in inputs
        .as_vec()
        .iter()
        .zip(outputs.as_vec())
        .map(|(m1, m2)| (m1[0], m1[1], m2[0]))
    {
        img.put_pixel(
            (x * width as f64) as u32,
            (y * width as f64) as u32,
            Luma([(v * 255.0) as u8]),
        );
    }

    img
}

fn inputs_to_retained_image(net: &NeuralNet, inputs: &Matrix2<f64>) -> RetainedImage {
    let mut buf = Vec::new();
    let outputs = net.run_batch(inputs).unwrap();
    let img = input_output_as_img(inputs, &outputs, 28);
    JpegEncoder::new(&mut buf).encode_image(&img).unwrap();
    RetainedImage::from_image_bytes("Output Image", &buf).unwrap()
}

fn generate_inputs(width: u32, third_dim: f64) -> Matrix2<f64> {
    let mut inputs = Vec::new();
    for x in 0..width {
        for y in 0..width {
            inputs.push(vec![
                x as f64 / width as f64,
                y as f64 / width as f64,
                third_dim,
            ]);
        }
    }

    Matrix2::from_vec(inputs).unwrap()
}

lazy_static! {
    static ref IMG1_INPUTS: Matrix2<f64> = get_normalized_training_data("assets/nine.jpg", 0.0).0;
    static ref IMG2_INPUTS: Matrix2<f64> = get_normalized_training_data("assets/six.jpg", 1.0).0;
    static ref IMG1_OUTPUTS: Matrix2<f64> = get_normalized_training_data("assets/nine.jpg", 0.0).1;
    static ref IMG2_OUTPUTS: Matrix2<f64> = get_normalized_training_data("assets/six.jpg", 1.0).1;
}

struct ImgGui {
    nn_gui: NNGui,
    target1: RetainedImage,
    target2: RetainedImage,
    slider_val: f64,
}

impl App for ImgGui {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        let (data, net) = self.nn_gui.get_data();

        if net.is_none() {
            return;
        }

        let net = net.unwrap();
        let data: Vec<_> = data.into_iter().map(|(i, j)| [i as f64, j]).collect();

        let output_image1 = inputs_to_retained_image(&net, &*IMG1_INPUTS);
        let output_image2 = inputs_to_retained_image(&net, &*IMG2_INPUTS);

        let slider_image = inputs_to_retained_image(&net, &generate_inputs(28, self.slider_val));

        CentralPanel::default().show(ctx, |ui| {
            ui.columns(2, |cols| {
                cols[0].vertical(|ui| {
                    let [i, j] = &data.last().unwrap_or(&[0.0, f64::INFINITY]);

                    ui.heading(format!("Iteration {} error: {j}", *i as usize));

                    Plot::new("Error").show(ui, |ui| ui.line(Line::new(data)));
                });

                cols[1].vertical(|ui| {
                    ui.horizontal(|ui| {
                        self.target1.show_scaled(ui, 5.0);
                        self.target2.show_scaled(ui, 5.0);
                    });

                    ui.horizontal(|ui| {
                        output_image1.show_scaled(ui, 5.0);
                        output_image2.show_scaled(ui, 5.0);
                    });

                    slider_image.show_scaled(ui, 10.0);

                    ui.add(Slider::new(&mut self.slider_val, 0.0..=1.0))
                });
            });
        });
    }
}

impl Visualizer for ImgGui {
    fn new(cc: &eframe::CreationContext, rx: Receiver<EpochState>) -> Self {
        let target_file = File::open("assets/nine.jpg").unwrap();
        let mut reader = BufReader::new(target_file);
        let mut buf = Vec::new();

        reader.read_to_end(&mut buf).unwrap();
        let target1 = RetainedImage::from_image_bytes("Target image nine", &buf).unwrap();

        let target_file = File::open("assets/six.jpg").unwrap();
        let mut reader = BufReader::new(target_file);
        let mut buf = Vec::new();

        reader.read_to_end(&mut buf).unwrap();
        let target2 = RetainedImage::from_image_bytes("Target image six", &buf).unwrap();

        Self {
            nn_gui: NNGui::new(cc, rx),
            target1,
            target2,
            slider_val: 0.0,
        }
    }
}

fn main() {
    let mut net = NeuralNet::new(3, 10, Activations::Sigmoid);
    net.add_layer(10, Activations::Sigmoid);
    net.add_layer(1, Activations::Sigmoid);

    let rate = 1.2;

    let optim = Optimizer::new(OptimizerMethod::Backprop, 10_000, rate)
        .with_log(Some(1))
        .with_batches(Some(75));

    let mut inputs = IMG1_INPUTS.clone();
    inputs.concat_rows(IMG2_INPUTS.clone()).unwrap();

    let mut targets = IMG1_OUTPUTS.clone();
    targets.concat_rows(IMG2_OUTPUTS.clone()).unwrap();

    let _ = optim.train_gui::<ImgGui>(&mut net, &inputs, &targets);
    // let _ = optim.train(&mut net, &INPUTS, &TARGETS);

    const WIDTH: u32 = 500;

    let img = input_output_as_img(
        &generate_inputs(WIDTH, 0.0),
        &net.run_batch(&inputs).unwrap(),
        WIDTH,
    );

    let _ = img.save("output.jpg");
}
