use std::{
    collections::VecDeque,
    sync::{mpsc::Receiver, Arc, Mutex},
    thread,
};

use eframe::CreationContext;
use egui::plot::{Line, Plot};

use crate::neural::NeuralNet;

/// State sent to the GUI renderer on each logged iteration of training
/// (iteration, error, copy of neural net)
pub type EpochState = (usize, f64, NeuralNet);

/// Any type that can be rendered and updated during training
pub trait Visualizer: eframe::App {
    const DATA_LIMIT: usize = 10_000;
    fn new(cc: &CreationContext, rx: Receiver<EpochState>) -> Self;
}

/// Default gui that displays error while training
pub struct NNGui {
    data: Arc<Mutex<VecDeque<(usize, f64)>>>,
    net: Arc<Mutex<Option<NeuralNet>>>,
}

impl Visualizer for NNGui {
    /// Initialize NNGui, but also start a thread that listens to a receiver and updates the state
    fn new(cc: &CreationContext, rx: Receiver<EpochState>) -> Self {
        let data = Arc::new(Mutex::new(VecDeque::new()));
        let net = Arc::new(Mutex::new(None));

        let data_clone = data.clone();
        let last_output_clone = net.clone();

        let ctx = cc.egui_ctx.clone();
        thread::spawn(move || loop {
            if let Ok((i, err, outs)) = rx.recv() {
                let mut data = data_clone.lock().unwrap();

                if data.len() == Self::DATA_LIMIT {
                    data.pop_front();
                }

                data.push_back((i, err));
                let _ = last_output_clone.lock().unwrap().insert(outs);

                ctx.request_repaint()
            }
        });

        Self { data, net }
    }
}

impl eframe::App for NNGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let data: Vec<_> = self
            .get_data()
            .0
            .into_iter()
            .map(|(i, j)| [i as f64, j])
            .collect();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                let [i, j] = data.last().unwrap_or(&[0.0, f64::INFINITY]);
                ui.heading(format!("Iteration {} error: {j}", *i as usize));
                Plot::new("Error").show(ui, |plot_ui| plot_ui.line(Line::new(data)));
            });
        });
    }
}

impl NNGui {
    /// Returns a clone of the data as a vec
    /// Blocks until it can get a lock on its state data
    pub fn get_data(&self) -> (Vec<(usize, f64)>, Option<NeuralNet>) {
        (
            self.data.lock().unwrap().clone().into(),
            self.net.lock().unwrap().clone(),
        )
    }
}
