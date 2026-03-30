// visionsort — a prediction-driven adaptive sorting algorithm
//
// Core thesis: each element touch does two jobs —
//   1. positions the element
//   2. updates the distribution model, reducing cost of all future touches
//
// Complexity: T(n, H) where H is the entropy of the input distribution.
// Cost per element decreases as the sort progresses.
//
// Optimizations in this version:
//   - Minrun (64): Phase 2 enforces minimum segment length, collapsing
//     random data from O(n) segments to O(n/64) segments.
//   - Bottom-up merge: Phase 5 uses pairwise merging instead of k-way
//     heap, which is more cache-friendly and has lower constant factors.
//   - Galloping merge: when one run dominates during merge, switch to
//     exponential search (Timsort's key trick for structured data).
//   - PlacementSort threshold lowered (>0.3) so larger segments on
//     low-entropy data actually route to the novel path.

#[cfg(target_arch = "wasm32")]
pub mod wasm;
#[cfg(target_arch = "wasm32")]
pub mod traced;

use std::cmp::Ordering;

const MINRUN: usize = 64;
const GALLOP_WIN: usize = 7; // consecutive wins before entering gallop mode

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
        if self.anchors.is_empty() { return (0.5, 0.0); }
        let min = self.estimated_min;
        let max = self.estimated_max;
        if (max - min).abs() < f64::EPSILON { return (0.5, 1.0); }
        let k = self.anchors.len();
        let pos = self.anchors.partition_point(|&a| a <= value);
        let predicted = if pos == 0 { 0.0 }
        else if pos >= k { 1.0 }
        else {
            let lo = self.anchors[pos - 1];
            let hi = self.anchors[pos];
            let anchor_frac = if (hi - lo).abs() < f64::EPSILON { 0.5 }
            else { (value - lo) / (hi - lo) };
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
// Sort Route
// ─────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq)]
pub enum SortRoute {
    NearlyFree,
    Verify,
    PlacementSort,
    FullSort,
    Trivial,
}

impl SortRoute {
    pub fn decide(disorder: f64, entropy: f64, len: usize) -> Self {
        // Trivial threshold raised to 2xMINRUN. Segments this small go
        // straight to insertion sort — scoring on disorder/entropy costs
        // more than it saves, especially on Clustered data with thousands
        // of tiny segments.
        if len <= MINRUN * 2 { return SortRoute::Trivial; }
        match (disorder > 0.3, entropy > 0.5) {
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

        self.model = self.phase1_glance(data);
        let segments = self.phase2_segment(data);
        let scored = self.phase3_disorder_map(data, segments);
        let mut model = self.model.clone();
        let sorted_ranges = self.phase4_fixate(data, scored, &mut model);
        self.model = model;
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
    // Phase 2 — Segmentation with Minrun
    //
    // Enforces a minimum segment length of MINRUN (64).
    // If a natural run is shorter, extend it with insertion sort
    // until it reaches MINRUN or the end of the array.
    // This is the key fix: random data produces O(n) tiny segments
    // without minrun, and O(n/MINRUN) segments with it.
    // ────────────────────────────────────────────────
    fn phase2_segment(&self, data: &mut [T]) -> Vec<Segment> {
        let n = data.len();
        let mut segments = Vec::new();
        let mut i = 0;

        while i < n {
            let seg_start = i;

            if i + 1 >= n {
                segments.push(Segment {
                    start: seg_start, end: n,
                    disorder: 0.0, entropy: 1.0, route: SortRoute::Trivial,
                });
                break;
            }

            // Detect run direction and walk to its natural end
            if data[i] > data[i + 1] {
                while i + 1 < n && data[i] >= data[i + 1] { i += 1; }
                i += 1;
                data[seg_start..i].reverse();
            } else {
                while i + 1 < n && data[i] <= data[i + 1] { i += 1; }
                i += 1;
            }

            // Enforce minrun: extend short runs with insertion sort
            let run_end = (seg_start + MINRUN).min(n);
            if i < run_end {
                // Extend to minrun length using insertion sort
                Self::insertion_sort(&mut data[seg_start..run_end]);
                i = run_end;
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
    fn phase3_disorder_map(&self, data: &[T], segments: Vec<Segment>) -> Vec<Segment> {
        let mut scored: Vec<Segment> = segments.into_iter().map(|mut seg| {
            let slice = &data[seg.start..seg.end];
            seg.disorder = Self::estimate_disorder(slice);
            let floats: Vec<f64> = slice.iter().map(|x| (*x).into()).collect();
            seg.entropy = DistributionModel::local_entropy(&floats);
            seg.route = SortRoute::decide(seg.disorder, seg.entropy, seg.len());
            seg
        }).collect();

        // Sort by priority — highest disorder × entropy × length first
        scored.sort_unstable_by(|a, b| {
            b.priority().partial_cmp(&a.priority()).unwrap_or(Ordering::Equal)
        });
        scored
    }

    // ────────────────────────────────────────────────
    // Phase 4 — Directed Fixation (bucket-local PlacementSort)
    // ────────────────────────────────────────────────
    fn phase4_fixate(
        &self,
        data: &mut [T],
        scored: Vec<Segment>,
        model: &mut DistributionModel,
    ) -> Vec<(usize, usize)> {
        let n = data.len();
        let mut sorted_ranges: Vec<(usize, usize)> = Vec::new();

        for seg in scored {
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
                    let b = ((len as f64).sqrt().ceil() as usize).max(1);
                    let mut buckets: Vec<Vec<T>> = vec![Vec::new(); b];
                    let values: Vec<T> = data[seg.start..seg.end].to_vec();

                    for &val in &values {
                        let (predicted_pos, _) = model.predict(val.into());
                        let bucket_idx = ((predicted_pos * (b - 1) as f64).round() as usize).min(b - 1);
                        buckets[bucket_idx].push(val);
                    }

                    for bucket in buckets.iter_mut() {
                        Self::insertion_sort(bucket);
                    }

                    let mut write_pos = seg.start;
                    for bucket in &buckets {
                        for &val in bucket {
                            data[write_pos] = val;
                            model.update(val.into(), write_pos, n);
                            write_pos += 1;
                        }
                    }

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
    // Phase 5 — Bottom-up merge with galloping
    //
    // Instead of k-way heap merge, use pairwise bottom-up merging:
    // merge adjacent pairs, then merge the results, doubling the
    // merge width each pass. O(n log k) but with much better cache
    // behaviour and no heap overhead.
    //
    // Galloping: when one run wins GALLOP_WIN times consecutively,
    // switch to exponential search to find how many more elements
    // from that run can be taken without checking the other.
    // Critical for nearly-sorted data where one run dominates.
    // ────────────────────────────────────────────────
    fn phase5_integrate(&self, data: &mut [T], mut sorted_ranges: Vec<(usize, usize)>) {
        let n = data.len();
        if sorted_ranges.is_empty() { return; }

        // Re-sort by start position — Phase 4 processes segments in
        // priority order (highest-disorder first) so they arrive here
        // out of array order. Bottom-up merge requires adjacency.
        sorted_ranges.sort_by_key(|&(s, _)| s);

        // Single segment covering the whole array — nothing to merge
        if sorted_ranges.len() == 1 {
            let (s, e) = sorted_ranges[0];
            if s == 0 && e == n { return; }
        }

        // Bottom-up pairwise merge with galloping.
        // Each pass merges adjacent pairs, halving segment count.
        // O(n log k) total, no heap overhead, cache-friendly access.
        while sorted_ranges.len() > 1 {
            let mut next: Vec<(usize, usize)> = Vec::with_capacity(sorted_ranges.len() / 2 + 1);
            let mut i = 0;
            while i < sorted_ranges.len() {
                if i + 1 < sorted_ranges.len() {
                    let (s1, e1) = sorted_ranges[i];
                    let (s2, e2) = sorted_ranges[i + 1];
                    if e1 == s2 {
                        Self::gallop_merge(data, s1, e1, e2);
                        next.push((s1, e2));
                    } else {
                        // Gap between segments — keep separate, merge next pass
                        next.push((s1, e1));
                        next.push((s2, e2));
                    }
                    i += 2;
                } else {
                    next.push(sorted_ranges[i]);
                    i += 1;
                }
            }
            sorted_ranges = next;
        }
    }

    // ── Galloping merge ──────────────────────────────────────────────
    // Merges data[lo..mid] and data[mid..hi] in-place.
    // Uses a temporary buffer for the left half only — O(n/2) memory.
    // Gallop mode kicks in when one side wins GALLOP_WIN times in a row.
    fn gallop_merge(data: &mut [T], lo: usize, mid: usize, hi: usize) {
        if lo >= mid || mid >= hi { return; }

        // Fast path: already in order — no work needed
        if data[mid - 1] <= data[mid] { return; }

        // Copy left half to temp buffer
        let left: Vec<T> = data[lo..mid].to_vec();
        let left_len = left.len();
        let right_len = hi - mid;

        let mut l = 0usize;  // index into left buffer
        let mut r = mid;     // index into data (right half)
        let mut out = lo;    // write position in data
        let mut left_wins = 0usize;
        let mut right_wins = 0usize;

        while l < left_len && r < hi {
            if left[l] <= data[r] {
                left_wins += 1;
                right_wins = 0;
                data[out] = left[l];
                l += 1;
                out += 1;

                // Enter left-gallop mode
                if left_wins >= GALLOP_WIN {
                    // Exponential search: how many left elements can we
                    // take before data[r] would win?
                    let mut step = 1usize;
                    let mut count = 0usize;
                    while l + step <= left_len && left[l + step - 1] <= data[r] {
                        count = step;
                        step *= 2;
                    }
                    // Refine with binary search
                    let upper = (l + step).min(left_len);
                    let gallop_count = left[l..upper]
                        .partition_point(|x| x <= &data[r]);
                    let take = gallop_count.max(count).min(left_len - l);
                    for k in 0..take {
                        data[out + k] = left[l + k];
                    }
                    l += take;
                    out += take;
                    left_wins = 0;
                }
            } else {
                right_wins += 1;
                left_wins = 0;
                data[out] = data[r];
                r += 1;
                out += 1;

                // Enter right-gallop mode
                if right_wins >= GALLOP_WIN {
                    // How many right elements can we take before left[l] wins?
                    let mut step = 1usize;
                    let mut count = 0usize;
                    while r + step <= hi && data[r + step - 1] < left[l] {
                        count = step;
                        step *= 2;
                    }
                    let upper = (r + step).min(hi);
                    let gallop_count = data[r..upper]
                        .partition_point(|x| x < &left[l]);
                    let take = gallop_count.max(count).min(hi - r);
                    let src_start = r;
                    for k in 0..take {
                        data[out + k] = data[src_start + k];
                    }
                    r += take;
                    out += take;
                    right_wins = 0;
                }
            }
        }

        // Drain remaining left elements
        while l < left_len {
            data[out] = left[l];
            l += 1;
            out += 1;
        }
        // Right elements are already in place
        let _ = right_len;
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
        let mut lt = 0; let mut gt = n - 1; let mut i = 0;
        while i < gt {
            if data[i] < data[gt] { data.swap(i, lt); lt += 1; i += 1; }
            else if data[i] > data[gt] { gt -= 1; data.swap(i, gt); }
            else { i += 1; }
        }
        (lt, gt)
    }

    fn heapsort(data: &mut [T]) {
        let n = data.len();
        for i in (0..n / 2).rev() { Self::sift_down(data, i, n); }
        for end in (1..n).rev() { data.swap(0, end); Self::sift_down(data, 0, end); }
    }

    fn sift_down(data: &mut [T], mut root: usize, end: usize) {
        loop {
            let mut largest = root;
            let left = 2 * root + 1; let right = 2 * root + 2;
            if left < end && data[left] > data[largest] { largest = left; }
            if right < end && data[right] > data[largest] { largest = right; }
            if largest == root { break; }
            data.swap(root, largest); root = largest;
        }
    }

    fn estimate_disorder(data: &[T]) -> f64 {
        let n = data.len();
        if n <= 1 { return 0.0; }
        let samples = (n as f64).sqrt() as usize + 1;
        let step = (n / samples).max(1);
        let mut inversions = 0usize; let mut pairs = 0usize;
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

    #[test] fn test_empty() { let mut d: Vec<f64>=vec![]; vision_sort(&mut d); assert_eq!(d,vec![]); }
    #[test] fn test_single() { let mut d=vec![1.0_f64]; vision_sort(&mut d); assert_eq!(d,vec![1.0]); }
    #[test] fn test_already_sorted() { let mut d=vec![1.0_f64,2.0,3.0,4.0,5.0]; vision_sort(&mut d); assert_eq!(d,vec![1.0,2.0,3.0,4.0,5.0]); }
    #[test] fn test_reverse_sorted() { let mut d=vec![5.0_f64,4.0,3.0,2.0,1.0]; vision_sort(&mut d); assert_eq!(d,vec![1.0,2.0,3.0,4.0,5.0]); }
    #[test] fn test_random_small() { let mut d=vec![3.0_f64,1.0,4.0,1.0,5.0,9.0,2.0,6.0]; let e=std_sorted(d.clone()); vision_sort(&mut d); assert_eq!(d,e); }
    #[test] fn test_nearly_sorted() { let mut d=vec![1.0_f64,2.0,4.0,3.0,5.0,6.0,7.0,8.0]; let e=std_sorted(d.clone()); vision_sort(&mut d); assert_eq!(d,e); }
    #[test] fn test_all_same() { let mut d=vec![7.0_f64;100]; vision_sort(&mut d); assert!(is_sorted(&d)); }
    #[test] fn test_two_elements() { let mut d=vec![2.0_f64,1.0]; vision_sort(&mut d); assert_eq!(d,vec![1.0,2.0]); }
    #[test] fn test_route_decision() {
        // Threshold is now 2*MINRUN = 128, use 200 for non-trivial routing
        assert_eq!(SortRoute::decide(0.1,0.1,200),SortRoute::NearlyFree);
        assert_eq!(SortRoute::decide(0.1,0.9,200),SortRoute::Verify);
        assert_eq!(SortRoute::decide(0.9,0.1,200),SortRoute::PlacementSort);
        assert_eq!(SortRoute::decide(0.9,0.9,200),SortRoute::FullSort);
        assert_eq!(SortRoute::decide(0.9,0.9,8),SortRoute::Trivial);
        assert_eq!(SortRoute::decide(0.1,0.1,100),SortRoute::Trivial); // 100 < 2*MINRUN
    }

    #[test]
    fn stress_random_1k() {
        let mut d: Vec<f64>=lcg_sequence(1000,42).iter().map(|&x|x as f64).collect();
        let e=std_sorted(d.clone()); vision_sort(&mut d); assert_eq!(d,e,"random 1k");
    }
    #[test]
    fn stress_random_10k() {
        let mut d: Vec<f64>=lcg_sequence(10_000,99).iter().map(|&x|x as f64).collect();
        let e=std_sorted(d.clone()); vision_sort(&mut d); assert_eq!(d,e,"random 10k");
    }
    #[test]
    fn stress_nearly_sorted_10k() {
        let mut d: Vec<f64>=(0..10_000).map(|x|x as f64).collect();
        let sw=lcg_sequence(200,7);
        for i in (0..sw.len()).step_by(2){let a=sw[i]%10_000;let b=sw[i+1]%10_000;d.swap(a,b);}
        let e=std_sorted(d.clone()); vision_sort(&mut d); assert_eq!(d,e,"nearly sorted 10k");
    }
    #[test]
    fn stress_reversed_10k() {
        let mut d: Vec<f64>=(0..10_000).map(|x|(10_000-x) as f64).collect();
        let e=std_sorted(d.clone()); vision_sort(&mut d); assert_eq!(d,e,"reversed 10k");
    }
    #[test]
    fn stress_clustered_10k() {
        let bands=[100.0_f64,200.0,300.0,400.0,500.0];
        let seq=lcg_sequence(10_000,13);
        let mut d: Vec<f64>=seq.iter().enumerate().map(|(i,&x)|bands[i%5]+(x%10) as f64).collect();
        let e=std_sorted(d.clone()); vision_sort(&mut d); assert_eq!(d,e,"clustered 10k");
    }
    #[test]
    fn stress_duplicates_10k() {
        let seq=lcg_sequence(10_000,55);
        let mut d: Vec<f64>=seq.iter().map(|&x|(x%100) as f64).collect();
        let e=std_sorted(d.clone()); vision_sort(&mut d); assert_eq!(d,e,"duplicates 10k");
    }
    #[test]
    fn cost_curve_entropy_decreases() {
        let data: Vec<f64>=lcg_sequence(1000,42).iter().map(|&x|x as f64).collect();
        let mut sorter=VisionSort::new();
        sorter.model=sorter.phase1_glance(&data);
        let initial=sorter.model.entropy;
        let n=data.len();
        let samples=lcg_sequence(100,7);
        for (i,&s) in samples.iter().enumerate(){let val=data[s%n] as f64;sorter.model.update(val,i,n);}
        assert!(sorter.model.entropy<=initial,"entropy should decrease: {}->{}",initial,sorter.model.entropy);
    }

    fn lcg_sequence(n: usize, seed: u64) -> Vec<usize> {
        let mut state=seed;
        (0..n).map(|_|{state=state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);(state>>33) as usize}).collect()
    }
}
