use wasm_bindgen::prelude::*;
use crate::{vision_sort, DistributionModel, VisionSort};

// ─────────────────────────────────────────────
// wasm_vision_sort — sort an array of f64
// ─────────────────────────────────────────────
#[wasm_bindgen]
pub fn wasm_vision_sort(data: Vec<f64>) -> Vec<f64> {
    let mut arr = data;
    vision_sort(&mut arr);
    arr
}

// ─────────────────────────────────────────────
// wasm_vision_sort_with_trace — sort + record
// entropy at each model update for the cost curve
// ─────────────────────────────────────────────
#[wasm_bindgen]
pub fn wasm_vision_sort_with_trace(data: Vec<f64>) -> JsValue {
    let mut arr = data.clone();
    let n = arr.len();

    // Run the real sort and capture the model state
    let mut sorter = VisionSort::<f64>::new();
    sorter.sort(&mut arr);

    // Now simulate the model updates on the original data to capture
    // the entropy trace (the sort itself doesn't expose per-step entropy)
    let mut model = DistributionModel::empty();

    // Seed from the glance phase
    let k = ((n as f64).log2().ceil() as usize).max(2);
    let mut samples: Vec<f64> = (0..k)
        .map(|i| {
            let idx = (i * (n - 1)) / (k - 1);
            data[idx]
        })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    model.estimated_min = samples[0];
    model.estimated_max = *samples.last().unwrap();
    model.anchors = samples.clone();
    model.entropy = DistributionModel::local_entropy(&samples);

    // Record entropy after each update
    let mut entropy_trace: Vec<f64> = Vec::with_capacity(n);
    entropy_trace.push(model.entropy);

    for (i, &val) in data.iter().enumerate() {
        // Use the sorted position as the actual position
        let actual_pos = arr.iter().position(|&x| x == val).unwrap_or(i);
        model.update(val, actual_pos, n);
        entropy_trace.push(model.entropy);
    }

    // Return as a JS object: { sorted: [...], entropyTrace: [...], finalEntropy, observations, anchorCount }
    let result = TraceResult {
        sorted: arr,
        entropy_trace,
        final_entropy: model.entropy,
        observations: model.observations,
        anchor_count: model.anchors.len(),
    };

    serde_wasm_bindgen::to_value(&result).unwrap()
}

#[derive(serde::Serialize)]
struct TraceResult {
    sorted: Vec<f64>,
    entropy_trace: Vec<f64>,
    final_entropy: f64,
    observations: usize,
    anchor_count: usize,
}

// ─────────────────────────────────────────────
// WasmModel — JS-accessible DistributionModel
// ─────────────────────────────────────────────
#[wasm_bindgen]
pub struct WasmModel {
    inner: DistributionModel,
}

#[wasm_bindgen]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: DistributionModel::empty(),
        }
    }

    pub fn predict(&self, value: f64) -> f64 {
        let (predicted, _) = self.inner.predict(value);
        predicted
    }

    pub fn predict_confidence(&self, value: f64) -> f64 {
        let (_, confidence) = self.inner.predict(value);
        confidence
    }

    pub fn update(&mut self, value: f64, pos: usize, n: usize) {
        self.inner.update(value, pos, n);
    }

    pub fn entropy(&self) -> f64 {
        self.inner.entropy
    }

    pub fn observations(&self) -> usize {
        self.inner.observations
    }

    pub fn anchor_count(&self) -> usize {
        self.inner.anchors.len()
    }

    pub fn estimated_min(&self) -> f64 {
        self.inner.estimated_min
    }

    pub fn estimated_max(&self) -> f64 {
        self.inner.estimated_max
    }

    /// Seed the model with initial samples (like Phase 1 — The Glance)
    pub fn seed(&mut self, samples: Vec<f64>) {
        let mut s = samples;
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if !s.is_empty() {
            self.inner.estimated_min = s[0];
            self.inner.estimated_max = *s.last().unwrap();
        }
        self.inner.entropy = DistributionModel::local_entropy(&s);
        self.inner.anchors = s;
    }
}
