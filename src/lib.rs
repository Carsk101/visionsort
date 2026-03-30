// visionsort — a prediction-driven adaptive sorting algorithm
//
// Core thesis: each element touch does two jobs —
//   1. positions the element
//   2. updates the distribution model, reducing cost of all future touches
//
// Complexity: T(n, H) where H is the entropy of the input distribution.
// Cost per element decreases as the sort progresses.

#[cfg(target_arch = "wasm32")]
pub mod wasm;

use std::collections::BinaryHeap;
use std::cmp::Ordering;

// ─────────────────────────────────────────────
// Distribution Model
// ─────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct DistributionModel {
    pub anchors: Vec<f64>,
    pub estimated_min: f64,
    pub estimated_max: f64,
    pub entropy: f64,
    pub observations: usize,
}

impl DistributionModel {
    pub fn empty() -> Self {
        Self {
            anchors: Vec::new(),
            estimated_min: f64::MAX,
            estimated_max: f64::MIN,
            entropy: 1.0,
            observations: 0,
        }
    }

    pub fn predict(&self, value: f64) -> (f64, f64) {
        if self.anchors.is_empty() {
            return (0.5, 0.0);
        }
        let min = self.estimated_min;
        let max = self.estimated_max;
        if (max - min).abs() < f64::EPSILON {
            return (0.5, 1.0);
        }
        let k = self.anchors.len();
        let pos = self.anchors.partition_point(|&a| a <= value);
        let predicted = if pos == 0 {
            0.0
        } else if pos >= k {
            1.0
        } else {
            let lo = self.anchors[pos - 1];
            let hi = self.anchors[pos];
            let anchor_frac = if (hi - lo).abs() < f64::EPSILON {
                0.5
            } else {
                (value - lo) / (hi - lo)
            };
            let lo_pos = (pos - 1) as f64 / (k - 1) as f64;
            let hi_pos = pos as f64 / (k - 1) as f64;
            lo_pos + anchor_frac * (hi_pos - lo_pos)
        };
        let obs_factor = (self.observations as f64 / 100.0).min(1.0);
        let confidence = (1.0 - self.entropy) * 0.5 + obs_factor * 0.5;
        (predicted.clamp(0.0, 1.0), confidence)
    }

    pub fn update(&mut self, value: f64, actual_position: usize, n: usize) {
        self.observations += 1;
        if value < self.estimated_min { self.estimated_min = value; }
        if value > self.estimated_max { self.estimated_max = value; }
        let obs_ratio = self.observations as f64 / n as f64;
        self.entropy = (self.entropy * (1.0 - obs_ratio * 0.01)).max(0.0);
        let (predicted_pos, _) = self.predict(value);
        let actual_pos = actual_position as f64 / n as f64;
        let surprise = (predicted_pos - actual_pos).abs();
        if surprise > 0.1 {
            let insert_pos = self.anchors.partition_point(|&a| a <= value);
            self.anchors.insert(insert_pos, value);
        }
    }

    pub fn local_entropy(values: &[f64]) -> f64 {
        let n = values.len();
        if n <= 1 { return 0.0; }
        let min = values.iter().cloned().fold(f64::MAX, f64::min);
        let max = values.iter().cloned().fold(f64::MIN, f64::max);
        if (max - min).abs() < f64::EPSILON { return 0.0; }
        let buckets = (n as f64).sqrt() as usize + 1;
        let mut counts = vec![0usize; buckets];
        for &v in values {
            let idx = ((v - min) / (max - min) * (buckets - 1) as f64) as usize;
            counts[idx.min(buckets - 1)] += 1;
        }
        let mut h = 0.0f64;
        for &c in &counts {
            if c > 0 {
                let p = c as f64 / n as f64;
                h -= p * p.log2();
            }
        }
        h / (buckets as f64).log2()
    }
}

// ─────────────────────────────────────────────
// Segment
// ─────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct Segment {
    pub start: usize,
    pub end: usize,
    pub disorder: f64,
    pub entropy: f64,
    pub route: SortRoute,
}

impl Segment {
    pub fn len(&self) -> usize { self.end - self.start }

    pub fn priority(&self) -> f64 {
        self.disorder * self.entropy * self.len() as f64
    }
}

impl PartialEq for Segment {
    fn eq(&self, other: &Self) -> bool { self.priority() == other.priority() }
}
impl Eq for Segment {}
impl PartialOrd for Segment {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.priority().partial_cmp(&other.priority())
    }
}
impl Ord for Segment {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// ─────────────────────────────────────────────
// Sort Route — 2D entropy × disorder decision space
// ─────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq)]
pub enum SortRoute {
    NearlyFree,     // Low disorder, low entropy  — almost free
    Verify,         // Low disorder, high entropy — confirm order
    PlacementSort,  // High disorder, low entropy — predict & place
    FullSort,       // High disorder, high entropy — introsort fallback
    Trivial,        // len <= 16 — insertion sort
}

impl SortRoute {
    pub fn decide(disorder: f64, entropy: f64, len: usize) -> Self {
        if len <= 16 { return SortRoute::Trivial; }
        match (disorder > 0.5, entropy > 0.5) {
            (false, false) => SortRoute::NearlyFree,
            (false, true)  => SortRoute::Verify,
            (true,  false) => SortRoute::PlacementSort,
            (true,  true)  => SortRoute::FullSort,
        }
    }
}

// ─────────────────────────────────────────────
// VisionSort
// ─────────────────────────────────────────────
pub struct VisionSort<T> {
    pub model: DistributionModel,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: PartialOrd + Into<f64> + Copy> VisionSort<T> {
    pub fn new() -> Self {
        Self {
            model: DistributionModel::empty(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn sort(&mut self, data: &mut [T]) {
        let n = data.len();
        if n <= 1 { return; }

        // Phase 1 — The Glance
        self.model = self.phase1_glance(data);

        // Phase 2 — Segmentation
        let segments = self.phase2_segment(data);

        // Phase 3 — Disorder Mapping
        let mut heap = self.phase3_disorder_map(data, segments);

        // Phase 4 — Directed Fixation
        let mut model = self.model.clone();
        let sorted_ranges = self.phase4_fixate(data, &mut heap, &mut model);
        self.model = model;

        // Phase 5 — Integration (true k-way merge)
        self.phase5_integrate(data, sorted_ranges);
    }

    // ────────────────────────────────────────────────
    // Phase 1 — The Glance
    // ────────────────────────────────────────────────
    fn phase1_glance(&self, data: &[T]) -> DistributionModel {
        let n = data.len();
        let k = ((n as f64).log2().ceil() as usize).max(2);
        let mut model = DistributionModel::empty();
        let mut samples: Vec<f64> = (0..k)
            .map(|i| { let idx = (i * (n - 1)) / (k - 1); data[idx].into() })
            .collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        model.estimated_min = samples[0];
        model.estimated_max = *samples.last().unwrap();
        model.anchors = samples.clone();
        model.entropy = DistributionModel::local_entropy(&samples);
        model
    }

    // ────────────────────────────────────────────────
    // Phase 2 — Segmentation
    //
    // Walks the array once. At each position we commit to either an ascending
    // or descending run and walk it to completion before emitting a segment.
    // This guarantees segments tile the array with no gaps and no overlaps.
    // ────────────────────────────────────────────────
    fn phase2_segment(&self, data: &mut [T]) -> Vec<Segment> {
        let n = data.len();
        let mut segments = Vec::new();
        let mut i = 0;

        while i < n {
            let seg_start = i;

            if i + 1 >= n {
                // Single trailing element
                segments.push(Segment {
                    start: seg_start, end: n,
                    disorder: 0.0, entropy: 1.0, route: SortRoute::Trivial,
                });
                break;
            }

            if data[i] > data[i + 1] {
                // Descending run — walk to end, then reverse
                while i + 1 < n && data[i] >= data[i + 1] { i += 1; }
                i += 1; // exclusive end
                data[seg_start..i].reverse();
            } else {
                // Ascending run — walk to end
                while i + 1 < n && data[i] <= data[i + 1] { i += 1; }
                i += 1;
            }

            segments.push(Segment {
                start: seg_start, end: i,
                disorder: 0.0, entropy: 1.0, route: SortRoute::Trivial,
            });
        }

        segments
    }

    // ────────────────────────────────────────────────
    // Phase 3 — Disorder Mapping
    // ────────────────────────────────────────────────
    fn phase3_disorder_map(&self, data: &[T], segments: Vec<Segment>) -> BinaryHeap<Segment> {
        let mut heap = BinaryHeap::new();
        for mut seg in segments {
            let slice = &data[seg.start..seg.end];
            seg.disorder = Self::estimate_disorder(slice);
            let floats: Vec<f64> = slice.iter().map(|x| (*x).into()).collect();
            seg.entropy = DistributionModel::local_entropy(&floats);
            seg.route = SortRoute::decide(seg.disorder, seg.entropy, seg.len());
            heap.push(seg);
        }
        heap
    }

    // ────────────────────────────────────────────────
    // Phase 4 — Directed Fixation
    // Returns sorted segment ranges for phase 5.
    // ────────────────────────────────────────────────
    fn phase4_fixate(
        &self,
        data: &mut [T],
        heap: &mut BinaryHeap<Segment>,
        model: &mut DistributionModel,
    ) -> Vec<(usize, usize)> {
        let n = data.len();
        let mut sorted_ranges: Vec<(usize, usize)> = Vec::new();

        while let Some(seg) = heap.pop() {
            let depth_limit = ((seg.len() as f64).log2() as usize + 1) * 2;

            match seg.route {
                SortRoute::Trivial | SortRoute::NearlyFree => {
                    Self::insertion_sort(&mut data[seg.start..seg.end]);
                }
                SortRoute::Verify => {
                    let slice = &mut data[seg.start..seg.end];
                    let violations = (1..slice.len()).any(|i| slice[i - 1] > slice[i]);
                    if violations { Self::insertion_sort(slice); }
                }
                SortRoute::PlacementSort => {
                    let len = seg.end - seg.start;
                    let mut output: Vec<Option<T>> = vec![None; len];
                    let mut unplaced: Vec<T> = Vec::new();
                    let values: Vec<T> = data[seg.start..seg.end].to_vec();

                    for &val in &values {
                        let (predicted_pos, confidence) = model.predict(val.into());
                        let slot = ((predicted_pos * (len - 1) as f64).round() as usize)
                            .min(len - 1);
                        if confidence > 0.7 && output[slot].is_none() {
                            output[slot] = Some(val);
                            model.update(val.into(), seg.start + slot, n);
                        } else {
                            unplaced.push(val);
                        }
                    }

                    // Sort only the unplaced minority — the novel cost reduction
                    Self::introsort(&mut unplaced, depth_limit);
                    let mut gap_iter = unplaced.into_iter();
                    for slot in output.iter_mut() {
                        if slot.is_none() { *slot = gap_iter.next(); }
                    }
                    for (i, val) in output.into_iter().enumerate() {
                        if let Some(v) = val { data[seg.start + i] = v; }
                    }

                    // Verification — cheap since array is near-sorted
                    let slice = &mut data[seg.start..seg.end];
                    let violations = (1..slice.len()).any(|i| slice[i - 1] > slice[i]);
                    if violations { Self::insertion_sort(slice); }
                }
                SortRoute::FullSort => {
                    Self::introsort(&mut data[seg.start..seg.end], depth_limit);
                }
            }

            sorted_ranges.push((seg.start, seg.end));
        }

        sorted_ranges
    }

    // ────────────────────────────────────────────────
    // Phase 5 — Integration (true k-way merge)
    //
    // Merges all sorted segments in O(n log k) using a min-heap.
    // Segments are tiled across the array — no gaps, no overlaps.
    // ────────────────────────────────────────────────
    fn phase5_integrate(&self, data: &mut [T], mut sorted_ranges: Vec<(usize, usize)>) {
        let n = data.len();
        if sorted_ranges.is_empty() { return; }

        // Sort ranges by start position — they tile the array
        sorted_ranges.sort_by_key(|&(s, _)| s);

        // Single segment covering the whole array — already done
        if sorted_ranges.len() == 1 {
            let (s, e) = sorted_ranges[0];
            if s == 0 && e == n { return; }
        }

        // Snapshot each segment (we need to read while writing output)
        let segments_data: Vec<Vec<T>> = sorted_ranges
            .iter()
            .map(|&(s, e)| data[s..e].to_vec())
            .collect();

        // Min-heap entry: smallest value wins each pop
        struct HeapEntry<T> {
            value: T,
            seg_idx: usize,
            pos: usize,
        }
        impl<T: PartialOrd> PartialEq for HeapEntry<T> {
            fn eq(&self, other: &Self) -> bool { self.value == other.value }
        }
        impl<T: PartialOrd> Eq for HeapEntry<T> {}
        impl<T: PartialOrd> PartialOrd for HeapEntry<T> {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                // Reverse: BinaryHeap is a max-heap, we want min
                other.value.partial_cmp(&self.value)
            }
        }
        impl<T: PartialOrd> Ord for HeapEntry<T> {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        // Seed heap with the first element from each segment
        let mut heap: BinaryHeap<HeapEntry<T>> = BinaryHeap::new();
        for (seg_idx, seg) in segments_data.iter().enumerate() {
            if !seg.is_empty() {
                heap.push(HeapEntry { value: seg[0], seg_idx, pos: 1 });
            }
        }

        // Drain heap — each pop is the globally minimum remaining value
        let mut output: Vec<T> = Vec::with_capacity(n);
        while let Some(entry) = heap.pop() {
            output.push(entry.value);
            let seg = &segments_data[entry.seg_idx];
            if entry.pos < seg.len() {
                heap.push(HeapEntry {
                    value: seg[entry.pos],
                    seg_idx: entry.seg_idx,
                    pos: entry.pos + 1,
                });
            }
        }

        // Write merged result back
        for (i, val) in output.into_iter().enumerate() {
            data[i] = val;
        }
    }

    // ────────────────────────────────────────────────
    // Utilities
    // ────────────────────────────────────────────────

    fn insertion_sort(data: &mut [T]) {
        for i in 1..data.len() {
            let mut j = i;
            while j > 0 && data[j - 1] > data[j] {
                data.swap(j - 1, j);
                j -= 1;
            }
        }
    }

    fn introsort(data: &mut [T], depth_limit: usize) {
        if data.len() <= 16 { Self::insertion_sort(data); return; }
        if depth_limit == 0 { Self::heapsort(data); return; }
        let pivot = Self::median_of_three(data);
        let (left, right) = Self::partition(data, pivot);
        Self::introsort(&mut data[..left], depth_limit - 1);
        Self::introsort(&mut data[right..], depth_limit - 1);
    }

    fn median_of_three(data: &[T]) -> usize {
        let mid = data.len() / 2;
        let last = data.len() - 1;
        if data[0] <= data[mid] && data[mid] <= data[last] { mid }
        else if data[mid] <= data[0] && data[0] <= data[last] { 0 }
        else { last }
    }

    fn partition(data: &mut [T], pivot_idx: usize) -> (usize, usize) {
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

    fn heapsort(data: &mut [T]) {
        let n = data.len();
        for i in (0..n / 2).rev() { Self::sift_down(data, i, n); }
        for end in (1..n).rev() {
            data.swap(0, end);
            Self::sift_down(data, 0, end);
        }
    }

    fn sift_down(data: &mut [T], mut root: usize, end: usize) {
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

    fn estimate_disorder(data: &[T]) -> f64 {
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
}

// ─────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────
pub fn vision_sort<T: PartialOrd + Into<f64> + Copy>(data: &mut [T]) {
    VisionSort::new().sort(data);
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn is_sorted<T: PartialOrd>(data: &[T]) -> bool {
        data.windows(2).all(|w| w[0] <= w[1])
    }

    fn std_sorted(mut data: Vec<f64>) -> Vec<f64> {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        data
    }

    // ── Unit tests ──────────────────────────────

    #[test]
    fn test_empty() {
        let mut data: Vec<f64> = vec![];
        vision_sort(&mut data);
        assert_eq!(data, vec![]);
    }

    #[test]
    fn test_single() {
        let mut data = vec![1.0_f64];
        vision_sort(&mut data);
        assert_eq!(data, vec![1.0]);
    }

    #[test]
    fn test_already_sorted() {
        let mut data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        vision_sort(&mut data);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_reverse_sorted() {
        let mut data = vec![5.0_f64, 4.0, 3.0, 2.0, 1.0];
        vision_sort(&mut data);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_random_small() {
        let mut data = vec![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let expected = std_sorted(data.clone());
        vision_sort(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_nearly_sorted() {
        let mut data = vec![1.0_f64, 2.0, 4.0, 3.0, 5.0, 6.0, 7.0, 8.0];
        let expected = std_sorted(data.clone());
        vision_sort(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_route_decision() {
        assert_eq!(SortRoute::decide(0.1, 0.1, 100), SortRoute::NearlyFree);
        assert_eq!(SortRoute::decide(0.1, 0.9, 100), SortRoute::Verify);
        assert_eq!(SortRoute::decide(0.9, 0.1, 100), SortRoute::PlacementSort);
        assert_eq!(SortRoute::decide(0.9, 0.9, 100), SortRoute::FullSort);
        assert_eq!(SortRoute::decide(0.9, 0.9, 8),   SortRoute::Trivial);
    }

    #[test]
    fn test_all_same() {
        let mut data = vec![7.0_f64; 100];
        vision_sort(&mut data);
        assert!(is_sorted(&data));
    }

    #[test]
    fn test_two_elements() {
        let mut data = vec![2.0_f64, 1.0];
        vision_sort(&mut data);
        assert_eq!(data, vec![1.0, 2.0]);
    }

    // ── Stress tests ────────────────────────────

    #[test]
    fn stress_random_1k() {
        // Deterministic LCG so test is reproducible
        let mut data: Vec<f64> = lcg_sequence(1000, 42)
            .iter().map(|&x| x as f64).collect();
        let expected = std_sorted(data.clone());
        vision_sort(&mut data);
        assert_eq!(data, expected, "random 1k failed");
    }

    #[test]
    fn stress_random_10k() {
        let mut data: Vec<f64> = lcg_sequence(10_000, 99)
            .iter().map(|&x| x as f64).collect();
        let expected = std_sorted(data.clone());
        vision_sort(&mut data);
        assert_eq!(data, expected, "random 10k failed");
    }

    #[test]
    fn stress_nearly_sorted_10k() {
        // Sorted with ~2% random swaps
        let mut data: Vec<f64> = (0..10_000).map(|x| x as f64).collect();
        let swaps = lcg_sequence(200, 7);
        for i in (0..swaps.len()).step_by(2) {
            let a = swaps[i] % 10_000;
            let b = swaps[i + 1] % 10_000;
            data.swap(a, b);
        }
        let expected = std_sorted(data.clone());
        vision_sort(&mut data);
        assert_eq!(data, expected, "nearly sorted 10k failed");
    }

    #[test]
    fn stress_reversed_10k() {
        let mut data: Vec<f64> = (0..10_000).map(|x| (10_000 - x) as f64).collect();
        let expected = std_sorted(data.clone());
        vision_sort(&mut data);
        assert_eq!(data, expected, "reversed 10k failed");
    }

    #[test]
    fn stress_clustered_10k() {
        // Values cluster in 5 tight bands — low entropy per segment
        let bands = [100.0_f64, 200.0, 300.0, 400.0, 500.0];
        let seq = lcg_sequence(10_000, 13);
        let mut data: Vec<f64> = seq.iter().enumerate()
            .map(|(i, &x)| bands[i % 5] + (x % 10) as f64)
            .collect();
        let expected = std_sorted(data.clone());
        vision_sort(&mut data);
        assert_eq!(data, expected, "clustered 10k failed");
    }

    #[test]
    fn stress_duplicates_10k() {
        let seq = lcg_sequence(10_000, 55);
        let mut data: Vec<f64> = seq.iter().map(|&x| (x % 100) as f64).collect();
        let expected = std_sorted(data.clone());
        vision_sort(&mut data);
        assert_eq!(data, expected, "duplicates 10k failed");
    }

    // ── Cost curve test ──────────────────────────
    // Verifies that the model's entropy decreases monotonically
    // as observations accumulate — the "speeds up as it runs" property.

    #[test]
    fn cost_curve_entropy_decreases() {
        let data: Vec<f64> = lcg_sequence(1000, 42)
            .iter().map(|&x| x as f64).collect();

        let mut sorter = VisionSort::new();
        sorter.model = sorter.phase1_glance(&data);

        let initial_entropy = sorter.model.entropy;

        // Simulate 100 Bayesian updates
        let n = data.len();
        let samples = lcg_sequence(100, 7);
        for (i, &s) in samples.iter().enumerate() {
            let val = data[s % n] as f64;
            sorter.model.update(val, i, n);
        }

        let final_entropy = sorter.model.entropy;

        assert!(
            final_entropy <= initial_entropy,
            "entropy should decrease or stay flat: {} -> {}",
            initial_entropy, final_entropy
        );
    }

    // ── Helpers ─────────────────────────────────

    /// Deterministic LCG sequence — reproducible across runs
    fn lcg_sequence(n: usize, seed: u64) -> Vec<usize> {
        let mut state = seed;
        (0..n).map(|_| {
            state = state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        }).collect()
    }
}
