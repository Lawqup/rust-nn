use std::{
    collections::VecDeque,
    sync::{mpsc::Receiver, Arc, Mutex},
    thread,
};

use eframe::CreationContext;
use egui::plot::{Line, Plot};

pub struct App {
    data: Arc<Mutex<VecDeque<[f64; 2]>>>,
}

impl App {
    const DATA_LIMIT: usize = 20_000;
    pub fn new(cc: &CreationContext, rx: Receiver<[f64; 2]>) -> Self {
        let data = Arc::new(Mutex::new(VecDeque::new()));
        let data_clone = data.clone();

        let ctx = cc.egui_ctx.clone();
        thread::spawn(move || loop {
            if let Ok(x) = rx.recv() {
                let mut data = data_clone.lock().unwrap();

                if data.len() == Self::DATA_LIMIT {
                    data.pop_front();
                }

                data.push_back(x);
                ctx.request_repaint()
            }
        });

        Self { data }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        let data: Vec<_> = self.data.lock().unwrap().clone().into();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                let [i, j] = &data.last().unwrap_or(&[0.0, f64::INFINITY]);
                ui.heading(format!("Iteration {} error: {j}", *i as usize));
                Plot::new("Error").show(ui, |plot_ui| plot_ui.line(Line::new(data)));
            });
        });
    }
}
