// traced.rs — instrumented sort that emits step-by-step snapshots
//
// This mirrors the real VisionSort algorithm but records array state
// at each meaningful moment for the step visualiser.

use serde::Serialize;
use crate::{DistributionModel, SortRoute};

#[derive(Serialize, Clone)]
pub struct SegmentInfo {
    pub start: usize,
    pub end: usize,
    pub route: String,
    pub disorder: f64,
    pub entropy: f64,
    pub active: bool,
}

#[derive(Serialize, Clone)]
pub struct SortStep {
    pub phase: String,
    pub label: String,
    pub array: Vec<f64>,
    pub highlights: Vec<usize>,
    pub segments: Vec<SegmentInfo>,
    pub entropy: f64,
}

pub fn traced_sort(data: &mut [f64]) -> Vec<SortStep> {
    let n = data.len();
    let mut steps: Vec<SortStep> = Vec::new();

    if n <= 1 {
        steps.push(SortStep {
            phase: "complete".into(),
            label: "Array is trivially sorted".into(),
            array: data.to_vec(),
            highlights: vec![],
            segments: vec![],
            entropy: 0.0,
        });
        return steps;
    }

    // ── Phase 1 — The Glance ──
    let k = ((n as f64).log2().ceil() as usize).max(2);
    let sample_indices: Vec<usize> = (0..k)
        .map(|i| (i * (n - 1)) / (k - 1))
        .collect();

    let mut model = DistributionModel::empty();
    let mut samples: Vec<f64> = sample_indices.iter().map(|&idx| data[idx]).collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    model.estimated_min = samples[0];
    model.estimated_max = *samples.last().unwrap();
    model.anchors = samples.clone();
    model.entropy = DistributionModel::local_entropy(&samples);

    steps.push(SortStep {
        phase: "glance".into(),
        label: format!("Phase 1 — The Glance: sampled {} anchor points, initial entropy {:.3}", k, model.entropy),
        array: data.to_vec(),
        highlights: sample_indices,
        segments: vec![],
        entropy: model.entropy,
    });

    // ── Phase 2 — Segmentation ──
    let mut segments_raw: Vec<(usize, usize, bool)> = Vec::new(); // (start, end, was_descending)
    let mut i = 0;
    while i < n {
        let seg_start = i;
        if i + 1 >= n {
            segments_raw.push((seg_start, n, false));
            break;
        }
        if data[i] > data[i + 1] {
            // Descending run
            while i + 1 < n && data[i] >= data[i + 1] { i += 1; }
            i += 1;
            data[seg_start..i].reverse();
            segments_raw.push((seg_start, i, true));
        } else {
            // Ascending run
            while i + 1 < n && data[i] <= data[i + 1] { i += 1; }
            i += 1;
            segments_raw.push((seg_start, i, false));
        }
    }

    let seg_infos: Vec<SegmentInfo> = segments_raw.iter().map(|&(s, e, _)| {
        SegmentInfo {
            start: s, end: e,
            route: "—".into(),
            disorder: 0.0, entropy: 0.0,
            active: false,
        }
    }).collect();

    steps.push(SortStep {
        phase: "segment".into(),
        label: format!("Phase 2 — Segmentation: found {} natural runs (descending runs reversed)", segments_raw.len()),
        array: data.to_vec(),
        highlights: vec![],
        segments: seg_infos,
        entropy: model.entropy,
    });

    // ── Phase 3 — Disorder Mapping ──
    struct ScoredSegment {
        start: usize,
        end: usize,
        disorder: f64,
        entropy: f64,
        route: SortRoute,
        priority: f64,
    }

    let mut scored_segments: Vec<ScoredSegment> = Vec::new();
    for &(s, e, _) in &segments_raw {
        let slice = &data[s..e];
        let disorder = estimate_disorder(slice);
        let floats: Vec<f64> = slice.to_vec();
        let seg_entropy = DistributionModel::local_entropy(&floats);
        let len = e - s;
        let route = SortRoute::decide(disorder, seg_entropy, len);
        let priority = disorder * seg_entropy * len as f64;
        scored_segments.push(ScoredSegment {
            start: s, end: e, disorder, entropy: seg_entropy, route, priority,
        });
    }

    let route_to_str = |r: &SortRoute| -> String {
        match r {
            SortRoute::NearlyFree => "NearlyFree".into(),
            SortRoute::Verify => "Verify".into(),
            SortRoute::PlacementSort => "PlacementSort".into(),
            SortRoute::FullSort => "FullSort".into(),
            SortRoute::Trivial => "Trivial".into(),
        }
    };

    let disorder_infos: Vec<SegmentInfo> = scored_segments.iter().map(|seg| {
        SegmentInfo {
            start: seg.start, end: seg.end,
            route: route_to_str(&seg.route),
            disorder: seg.disorder,
            entropy: seg.entropy,
            active: false,
        }
    }).collect();

    steps.push(SortStep {
        phase: "disorder".into(),
        label: format!("Phase 3 — Disorder Mapping: scored {} segments, assigned routes", scored_segments.len()),
        array: data.to_vec(),
        highlights: vec![],
        segments: disorder_infos.clone(),
        entropy: model.entropy,
    });

    // ── Phase 4 — Directed Fixation ──
    // Sort segments by priority (highest first)
    scored_segments.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

    let mut sorted_ranges: Vec<(usize, usize)> = Vec::new();

    for seg_idx in 0..scored_segments.len() {
        let seg = &scored_segments[seg_idx];
        let start = seg.start;
        let end = seg.end;
        let route = seg.route.clone();
        let depth_limit = (((end - start) as f64).log2() as usize + 1) * 2;

        match route {
            SortRoute::Trivial | SortRoute::NearlyFree => {
                insertion_sort(&mut data[start..end]);
            }
            SortRoute::Verify => {
                let slice = &mut data[start..end];
                let violations = (1..slice.len()).any(|i| slice[i - 1] > slice[i]);
                if violations { insertion_sort(slice); }
            }
            SortRoute::PlacementSort => {
                let len = end - start;
                let b = ((len as f64).sqrt().ceil() as usize).max(1);
                let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); b];

                let values: Vec<f64> = data[start..end].to_vec();
                for &val in &values {
                    let (predicted_pos, _) = model.predict(val);
                    let bucket_idx = ((predicted_pos * (b - 1) as f64).round() as usize).min(b - 1);
                    buckets[bucket_idx].push(val);
                }

                for bucket in buckets.iter_mut() {
                    insertion_sort(bucket);
                }

                let mut write_pos = start;
                for bucket in &buckets {
                    for &val in bucket {
                        data[write_pos] = val;
                        model.update(val, write_pos, n);
                        write_pos += 1;
                    }
                }

                let slice = &mut data[start..end];
                let violations = (1..slice.len()).any(|i| slice[i - 1] > slice[i]);
                if violations { insertion_sort(slice); }
            }
            SortRoute::FullSort => {
                introsort(&mut data[start..end], depth_limit);
            }
        }

        sorted_ranges.push((start, end));

        // Build segment infos with current segment marked active
        let step_segments: Vec<SegmentInfo> = scored_segments.iter().enumerate().map(|(i, s)| {
            SegmentInfo {
                start: s.start, end: s.end,
                route: route_to_str(&s.route),
                disorder: s.disorder,
                entropy: s.entropy,
                active: i == seg_idx,
            }
        }).collect();

        let highlight_range: Vec<usize> = (start..end).collect();

        steps.push(SortStep {
            phase: "fixate".into(),
            label: format!("Phase 4 — Fixation: sorted segment [{}..{}] via {}", start, end, route_to_str(&route)),
            array: data.to_vec(),
            highlights: highlight_range,
            segments: step_segments,
            entropy: model.entropy,
        });
    }

    // ── Phase 5 — Integration ──
    sorted_ranges.sort_by_key(|&(s, _)| s);

    if sorted_ranges.len() == 1 {
        let (s, e) = sorted_ranges[0];
        if s == 0 && e == n {} // do nothing
    } else if sorted_ranges.len() > 1 {
        let scratch: Vec<f64> = data.to_vec();

        // Min-heap merge
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        struct HeapEntry {
            value: f64,
            pos: usize,
            end: usize,
        }
        impl PartialEq for HeapEntry {
            fn eq(&self, other: &Self) -> bool { self.value == other.value }
        }
        impl Eq for HeapEntry {}
        impl PartialOrd for HeapEntry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                other.value.partial_cmp(&self.value)
            }
        }
        impl Ord for HeapEntry {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        for &(s, e) in &sorted_ranges {
            if s < e {
                heap.push(HeapEntry { value: scratch[s], pos: s + 1, end: e });
            }
        }

        let mut out = 0;
        while let Some(entry) = heap.pop() {
            data[out] = entry.value;
            out += 1;
            if entry.pos < entry.end {
                heap.push(HeapEntry {
                    value: scratch[entry.pos],
                    pos: entry.pos + 1,
                    end: entry.end,
                });
            }
        }
    }

    steps.push(SortStep {
        phase: "merge".into(),
        label: format!("Phase 5 — Integration: k-way merge of {} sorted segments complete", sorted_ranges.len()),
        array: data.to_vec(),
        highlights: vec![],
        segments: vec![],
        entropy: model.entropy,
    });

    steps
}

// ── Utility functions (standalone for f64) ──

fn insertion_sort(data: &mut [f64]) {
    for i in 1..data.len() {
        let mut j = i;
        while j > 0 && data[j - 1] > data[j] {
            data.swap(j - 1, j);
            j -= 1;
        }
    }
}

fn introsort(data: &mut [f64], depth_limit: usize) {
    if data.len() <= 16 { insertion_sort(data); return; }
    if depth_limit == 0 { heapsort(data); return; }
    let pivot = median_of_three(data);
    let (left, right) = partition(data, pivot);
    introsort(&mut data[..left], depth_limit - 1);
    introsort(&mut data[right..], depth_limit - 1);
}

fn median_of_three(data: &[f64]) -> usize {
    let mid = data.len() / 2;
    let last = data.len() - 1;
    if data[0] <= data[mid] && data[mid] <= data[last] { mid }
    else if data[mid] <= data[0] && data[0] <= data[last] { 0 }
    else { last }
}

fn partition(data: &mut [f64], pivot_idx: usize) -> (usize, usize) {
    let n = data.len();
    data.swap(pivot_idx, n - 1);
    let mut lt = 0;
    let mut gt = n - 1;
    let mut i = 0;
    while i < gt {
        if data[i] < data[gt] {
            data.swap(i, lt); lt += 1; i += 1;
        } else if data[i] > data[gt] {
            gt -= 1; data.swap(i, gt);
        } else {
            i += 1;
        }
    }
    (lt, gt)
}

fn heapsort(data: &mut [f64]) {
    let n = data.len();
    for i in (0..n / 2).rev() { sift_down(data, i, n); }
    for end in (1..n).rev() {
        data.swap(0, end);
        sift_down(data, 0, end);
    }
}

fn sift_down(data: &mut [f64], mut root: usize, end: usize) {
    loop {
        let mut largest = root;
        let left = 2 * root + 1;
        let right = 2 * root + 2;
        if left < end && data[left] > data[largest] { largest = left; }
        if right < end && data[right] > data[largest] { largest = right; }
        if largest == root { break; }
        data.swap(root, largest);
        root = largest;
    }
}

fn estimate_disorder(data: &[f64]) -> f64 {
    let n = data.len();
    if n <= 1 { return 0.0; }
    let samples = (n as f64).sqrt() as usize + 1;
    let step = (n / samples).max(1);
    let mut inversions = 0usize;
    let mut pairs = 0usize;
    for i in 0..samples {
        for j in (i + 1)..samples {
            let a = (i * step).min(n - 1);
            let b = (j * step).min(n - 1);
            if data[a] > data[b] { inversions += 1; }
            pairs += 1;
        }
    }
    if pairs == 0 { 0.0 } else { inversions as f64 / pairs as f64 }
}
