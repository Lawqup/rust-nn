use std::{
    collections::VecDeque,
    sync::{mpsc::Receiver, Arc, Mutex},
    thread,
};

use eframe::CreationContext;
use egui::plot::{Line, Plot};

use crate::matrix::Matrix2;

/// State sent to the GUI renderer on each logged iteration of training
/// (iteration, error, outputs)
pub type IterationState = (usize, f64, Matrix2<f64>);

/// Any type that can be rendered and updated during training
pub trait Visualizer: eframe::App + 'static {
    const DATA_LIMIT: usize = 10_000;
    fn new(cc: &CreationContext, rx: Receiver<IterationState>) -> Self;
}

/// Default gui that displays error while training
pub struct NNGui {
    data: Arc<Mutex<VecDeque<(usize, f64)>>>,
    last_output: Arc<Mutex<Option<Matrix2<f64>>>>,
}

impl Visualizer for NNGui {
    /// Initialize NNGui, but also start a thread that listens to a receiver and updates the state
    fn new(cc: &CreationContext, rx: Receiver<IterationState>) -> Self {
        let data = Arc::new(Mutex::new(VecDeque::new()));
        let last_output = Arc::new(Mutex::new(None));

        let data_clone = data.clone();
        let last_output_clone = last_output.clone();

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

        Self { data, last_output }
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
    pub fn get_data(&self) -> (Vec<(usize, f64)>, Option<Matrix2<f64>>) {
        (
            self.data.lock().unwrap().clone().into(),
            self.last_output.lock().unwrap().clone(),
        )
    }
}
