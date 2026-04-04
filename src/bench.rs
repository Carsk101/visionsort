// bench.rs — instrumented 200k element benchmark
// Captures per-phase timing, entropy trace, and segment routing data
// across all five input classes. Outputs JSON to stdout.

use std::time::Instant;
use visionsort::vision_sort;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

// ── Inline the algorithm types so we can instrument them ──
// We duplicate the necessary types here to add instrumentation hooks
// without modifying the public library API.

#[derive(Debug, Clone)]
struct DistributionModel {
    anchors: Vec<f64>,
    estimated_min: f64,
    estimated_max: f64,
    entropy: f64,
    observations: usize,
}

impl DistributionModel {
    fn empty() -> Self {
        Self {
            anchors: Vec::new(),
            estimated_min: f64::MAX,
            estimated_max: f64::MIN,
            entropy: 1.0,
            observations: 0,
        }
    }

    fn predict(&self, value: f64) -> (f64, f64) {
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

    fn update(&mut self, value: f64, actual_position: usize, n: usize) {
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

    fn local_entropy(values: &[f64]) -> f64 {
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

#[derive(Debug, Clone, PartialEq)]
enum SortRoute { NearlyFree, Verify, PlacementSort, FullSort, Trivial }

impl SortRoute {
    fn decide(disorder: f64, entropy: f64, len: usize) -> Self {
        if len <= 16 { return SortRoute::Trivial; }
        match (disorder > 0.5, entropy > 0.5) {
            (false, false) => SortRoute::NearlyFree,
            (false, true)  => SortRoute::Verify,
            (true,  false) => SortRoute::PlacementSort,
            (true,  true)  => SortRoute::FullSort,
        }
    }
    fn name(&self) -> &str {
        match self {
            SortRoute::NearlyFree => "NearlyFree",
            SortRoute::Verify => "Verify",
            SortRoute::PlacementSort => "PlacementSort",
            SortRoute::FullSort => "FullSort",
            SortRoute::Trivial => "Trivial",
        }
    }
}

// ── Instrumented run result ──
struct RunResult {
    input_type: String,
    n: usize,
    phase1_us: u64,
    phase2_us: u64,
    phase3_us: u64,
    phase4_us: u64,
    phase5_us: u64,
    total_us: u64,
    entropy_initial: f64,
    entropy_final: f64,
    entropy_trace: Vec<f64>,  // sampled at 100 points
    segment_count: usize,
    route_counts: [usize; 5], // Trivial, NearlyFree, Verify, PlacementSort, FullSort
    anchor_count_final: usize,
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

fn insertion_sort(data: &mut [f64]) {
    for i in 1..data.len() {
        let mut j = i;
        while j > 0 && data[j - 1] > data[j] { data.swap(j - 1, j); j -= 1; }
    }
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
    let mut lt = 0; let mut gt = n - 1; let mut i = 0;
    while i < gt {
        if data[i] < data[gt] { data.swap(i, lt); lt += 1; i += 1; }
        else if data[i] > data[gt] { gt -= 1; data.swap(i, gt); }
        else { i += 1; }
    }
    (lt, gt)
}

fn sift_down(data: &mut [f64], mut root: usize, end: usize) {
    loop {
        let mut largest = root;
        let left = 2 * root + 1; let right = 2 * root + 2;
        if left < end && data[left] > data[largest] { largest = left; }
        if right < end && data[right] > data[largest] { largest = right; }
        if largest == root { break; }
        data.swap(root, largest); root = largest;
    }
}

fn heapsort(data: &mut [f64]) {
    let n = data.len();
    for i in (0..n / 2).rev() { sift_down(data, i, n); }
    for end in (1..n).rev() { data.swap(0, end); sift_down(data, 0, end); }
}

fn introsort(data: &mut [f64], depth_limit: usize) {
    if data.len() <= 16 { insertion_sort(data); return; }
    if depth_limit == 0 { heapsort(data); return; }
    let pivot = median_of_three(data);
    let (left, right) = partition(data, pivot);
    introsort(&mut data[..left], depth_limit - 1);
    introsort(&mut data[right..], depth_limit - 1);
}

fn run_instrumented(input_type: &str, data: &mut Vec<f64>) -> RunResult {
    let n = data.len();
    let total_start = Instant::now();

    // ── Phase 1 ──
    let t1 = Instant::now();
    let k = ((n as f64).log2().ceil() as usize).max(2);
    let mut model = DistributionModel::empty();
    let mut samples: Vec<f64> = (0..k)
        .map(|i| { let idx = (i * (n - 1)) / (k - 1); data[idx] })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    model.estimated_min = samples[0];
    model.estimated_max = *samples.last().unwrap();
    model.anchors = samples.clone();
    model.entropy = DistributionModel::local_entropy(&samples);
    let entropy_initial = model.entropy;
    let phase1_us = t1.elapsed().as_micros() as u64;

    // ── Phase 2 ──
    let t2 = Instant::now();
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        let seg_start = i;
        if i + 1 >= n { segments.push((seg_start, n)); break; }
        if data[i] > data[i + 1] {
            while i + 1 < n && data[i] >= data[i + 1] { i += 1; }
            i += 1;
            data[seg_start..i].reverse();
        } else {
            while i + 1 < n && data[i] <= data[i + 1] { i += 1; }
            i += 1;
        }
        segments.push((seg_start, i));
    }
    let segment_count = segments.len();
    let phase2_us = t2.elapsed().as_micros() as u64;

    // ── Phase 3 ──
    let t3 = Instant::now();
    struct ScoredSeg { start: usize, end: usize, route: SortRoute, priority: f64 }
    let mut scored: Vec<ScoredSeg> = segments.iter().map(|&(s, e)| {
        let slice = &data[s..e];
        let disorder = estimate_disorder(slice);
        let ent = DistributionModel::local_entropy(slice);
        let len = e - s;
        let route = SortRoute::decide(disorder, ent, len);
        let priority = disorder * ent * len as f64;
        ScoredSeg { start: s, end: e, route, priority }
    }).collect();
    scored.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(Ordering::Equal));

    // Count routes
    let mut route_counts = [0usize; 5];
    for seg in &scored {
        let idx = match seg.route {
            SortRoute::Trivial => 0,
            SortRoute::NearlyFree => 1,
            SortRoute::Verify => 2,
            SortRoute::PlacementSort => 3,
            SortRoute::FullSort => 4,
        };
        route_counts[idx] += 1;
    }
    let phase3_us = t3.elapsed().as_micros() as u64;

    // ── Phase 4 ──
    let t4 = Instant::now();
    let sample_interval = (n / 100).max(1);
    let mut entropy_trace: Vec<f64> = vec![entropy_initial];
    let mut sorted_ranges: Vec<(usize, usize)> = Vec::new();
    let mut obs_counter = 0usize;

    for seg in &scored {
        let (start, end) = (seg.start, seg.end);
        let len = end - start;
        let depth_limit = ((len as f64).log2() as usize + 1) * 2;

        match seg.route {
            SortRoute::Trivial | SortRoute::NearlyFree => {
                insertion_sort(&mut data[start..end]);
            }
            SortRoute::Verify => {
                let slice = &mut data[start..end];
                let violations = (1..slice.len()).any(|i| slice[i-1] > slice[i]);
                if violations { insertion_sort(slice); }
            }
            SortRoute::PlacementSort => {
                let b = ((len as f64).sqrt().ceil() as usize).max(1);
                let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); b];
                let values: Vec<f64> = data[start..end].to_vec();
                for &val in &values {
                    let (predicted_pos, _) = model.predict(val);
                    let bucket_idx = ((predicted_pos * (b - 1) as f64).round() as usize).min(b - 1);
                    buckets[bucket_idx].push(val);
                }
                for bucket in buckets.iter_mut() { insertion_sort(bucket); }
                let mut write_pos = start;
                for bucket in &buckets {
                    for &val in bucket {
                        data[write_pos] = val;
                        model.update(val, write_pos, n);
                        obs_counter += 1;
                        if obs_counter % sample_interval == 0 {
                            entropy_trace.push(model.entropy);
                        }
                        write_pos += 1;
                    }
                }
                let slice = &mut data[start..end];
                let violations = (1..slice.len()).any(|i| slice[i-1] > slice[i]);
                if violations { insertion_sort(slice); }
            }
            SortRoute::FullSort => {
                introsort(&mut data[start..end], depth_limit);
            }
        }
        sorted_ranges.push((start, end));
    }
    let phase4_us = t4.elapsed().as_micros() as u64;

    let entropy_final = model.entropy;
    let anchor_count_final = model.anchors.len();

    // Ensure trace has at least 2 points
    if entropy_trace.len() < 2 { entropy_trace.push(entropy_final); }

    // ── Phase 5 ──
    let t5 = Instant::now();
    sorted_ranges.sort_by_key(|&(s, _)| s);
    if sorted_ranges.len() > 1 {
        let scratch: Vec<f64> = data.to_vec();
        struct HeapEntry { value: f64, pos: usize, end: usize }
        impl PartialEq for HeapEntry { fn eq(&self, o: &Self) -> bool { self.value == o.value } }
        impl Eq for HeapEntry {}
        impl PartialOrd for HeapEntry {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> { o.value.partial_cmp(&self.value) }
        }
        impl Ord for HeapEntry {
            fn cmp(&self, o: &Self) -> Ordering { self.partial_cmp(o).unwrap_or(Ordering::Equal) }
        }
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        for &(s, e) in &sorted_ranges {
            if s < e { heap.push(HeapEntry { value: scratch[s], pos: s + 1, end: e }); }
        }
        let mut out = 0;
        while let Some(entry) = heap.pop() {
            data[out] = entry.value; out += 1;
            if entry.pos < entry.end {
                heap.push(HeapEntry { value: scratch[entry.pos], pos: entry.pos + 1, end: entry.end });
            }
        }
    }
    let phase5_us = t5.elapsed().as_micros() as u64;
    let total_us = total_start.elapsed().as_micros() as u64;

    RunResult {
        input_type: input_type.to_string(),
        n,
        phase1_us, phase2_us, phase3_us, phase4_us, phase5_us, total_us,
        entropy_initial, entropy_final,
        entropy_trace,
        segment_count,
        route_counts,
        anchor_count_final,
    }
}

fn lcg_sequence(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / (u32::MAX as f64) * 1_000_000.0
    }).collect()
}

fn run_scale_benchmark(n: usize, label: &str) {
    let inputs: Vec<(&str, Vec<f64>)> = vec![
        ("Random",        lcg_sequence(n, 42)),
        ("Nearly Sorted", {
            let mut d: Vec<f64> = (0..n).map(|x| x as f64).collect();
            let swaps = lcg_sequence((n / 50).min(4000), 7);
            for i in (0..swaps.len()).step_by(2) {
                let a = (swaps[i] as usize) % n;
                let b = (swaps[i+1] as usize) % n;
                d.swap(a, b);
            }
            d
        }),
        ("Clustered",     {
            let bands = [100_000.0_f64, 200_000.0, 300_000.0, 400_000.0, 500_000.0];
            let seq = lcg_sequence(n, 13);
            seq.iter().enumerate().map(|(i, &x)| bands[i % 5] + x % 10_000.0).collect()
        }),
        ("Reversed",      (0..n).map(|x| (n - x) as f64).collect()),
        ("All Same",      vec![42.0_f64; n]),
    ];

    eprintln!("
=== {} — {} elements ===
", label, n);
    eprintln!("{:<16} {:>12} {:>12} {:>12} {:>12}",
        "Input", "VisionSort", "Quicksort", "std::sort", "Merge sort");
    eprintln!("{}", "-".repeat(68));

    for (name, data) in &inputs {
        let r = run_comparison(name, data);
        let vs_qs = if r.visionsort_us < r.quicksort_us {
            format!("+{}%", (r.quicksort_us - r.visionsort_us) * 100 / r.quicksort_us)
        } else {
            format!("-{}%", (r.visionsort_us - r.quicksort_us) * 100 / r.visionsort_us)
        };
        eprintln!("{:<16} {:>10}μs {:>10}μs {:>10}μs {:>10}μs  vs QS: {}",
            r.input_type, r.visionsort_us, r.quicksort_us, r.std_sort_us, r.merge_sort_us, vs_qs);
    }
}

fn main() {
    let n = 200_000;

    // Generate all input types
    let mut inputs: Vec<(&str, Vec<f64>)> = vec![
        ("Random",         lcg_sequence(n, 42)),
        ("Nearly Sorted",  {
            let mut d: Vec<f64> = (0..n).map(|x| x as f64).collect();
            let swaps = lcg_sequence(4000, 7);
            for i in (0..swaps.len()).step_by(2) {
                let a = (swaps[i] as usize) % n;
                let b = (swaps[i+1] as usize) % n;
                d.swap(a, b);
            }
            d
        }),
        ("Clustered",      {
            let bands = [100_000.0_f64, 200_000.0, 300_000.0, 400_000.0, 500_000.0];
            let seq = lcg_sequence(n, 13);
            seq.iter().enumerate().map(|(i, &x)| bands[i % 5] + x % 10_000.0).collect()
        }),
        ("Reversed",       (0..n).map(|x| (n - x) as f64).collect()),
        ("All Same",       vec![42.0_f64; n]),
    ];

    // Run all benchmarks
    let results: Vec<RunResult> = inputs.iter_mut()
        .map(|(name, data)| run_instrumented(name, data))
        .collect();

    // Verify all sorted correctly
    for (result, (_, data)) in results.iter().zip(inputs.iter()) {
        let sorted = data.windows(2).all(|w| w[0] <= w[1]);
        if !sorted {
            eprintln!("SORT FAILED for {}", result.input_type);
        }
    }

    // ── Comparison benchmark ──────────────────────────────────────
    // Generate fresh copies for fair comparison (no instrumentation overhead)
    let cmp_inputs: Vec<(&str, Vec<f64>)> = vec![
        ("Random",        lcg_sequence(n, 42)),
        ("Nearly Sorted", {
            let mut d: Vec<f64> = (0..n).map(|x| x as f64).collect();
            let swaps = lcg_sequence(4000, 7);
            for i in (0..swaps.len()).step_by(2) {
                let a = (swaps[i] as usize) % n;
                let b = (swaps[i+1] as usize) % n;
                d.swap(a, b);
            }
            d
        }),
        ("Clustered",     {
            let bands = [100_000.0_f64, 200_000.0, 300_000.0, 400_000.0, 500_000.0];
            let seq = lcg_sequence(n, 13);
            seq.iter().enumerate().map(|(i, &x)| bands[i % 5] + x % 10_000.0).collect()
        }),
        ("Reversed",      (0..n).map(|x| (n - x) as f64).collect()),
        ("All Same",      vec![42.0_f64; n]),
    ];

    let cmp_results: Vec<CompResult> = cmp_inputs.iter()
        .map(|(name, data)| run_comparison(name, data))
        .collect();

    // Print comparison table to stderr so JSON stdout stays clean
    eprintln!("\n=== SORT COMPARISON — {} elements ===\n", n);
    eprintln!("{:<16} {:>12} {:>12} {:>12} {:>12}",
        "Input", "VisionSort", "Quicksort", "std::sort", "Merge sort");
    eprintln!("{}", "-".repeat(68));
    for r in &cmp_results {
        eprintln!("{:<16} {:>10}μs {:>10}μs {:>10}μs {:>10}μs",
            r.input_type, r.visionsort_us, r.quicksort_us, r.std_sort_us, r.merge_sort_us);
    }
    eprintln!();

    // Also embed in JSON output
    // Output JSON
    println!("{{");
    println!("  \"n\": {},", n);
    println!("  \"results\": [");
    for (i, r) in results.iter().enumerate() {
        let comma = if i < results.len() - 1 { "," } else { "" };
        let trace_json: String = r.entropy_trace.iter()
            .map(|v| format!("{:.6}", v))
            .collect::<Vec<_>>()
            .join(",");
        println!("    {{");
        println!("      \"inputType\": \"{}\",", r.input_type);
        println!("      \"n\": {},", r.n);
        println!("      \"timing\": {{");
        println!("        \"phase1\": {},", r.phase1_us);
        println!("        \"phase2\": {},", r.phase2_us);
        println!("        \"phase3\": {},", r.phase3_us);
        println!("        \"phase4\": {},", r.phase4_us);
        println!("        \"phase5\": {},", r.phase5_us);
        println!("        \"total\": {}", r.total_us);
        println!("      }},");
        println!("      \"entropy\": {{");
        println!("        \"initial\": {:.6},", r.entropy_initial);
        println!("        \"final\": {:.6},", r.entropy_final);
        println!("        \"trace\": [{}]", trace_json);
        println!("      }},");
        println!("      \"segments\": {{");
        println!("        \"count\": {},", r.segment_count);
        println!("        \"routes\": {{");
        println!("          \"Trivial\": {},", r.route_counts[0]);
        println!("          \"NearlyFree\": {},", r.route_counts[1]);
        println!("          \"Verify\": {},", r.route_counts[2]);
        println!("          \"PlacementSort\": {},", r.route_counts[3]);
        println!("          \"FullSort\": {}", r.route_counts[4]);
        println!("        }}");
        println!("      }},");
        println!("      \"anchorCount\": {}", r.anchor_count_final);
        println!("    }}{}", comma);
    }
    println!("  ]");
    println!("}}");

    // Scale benchmarks — 1M and 10M elements
    run_scale_benchmark(1_000_000, "1M ELEMENTS");
    run_scale_benchmark(10_000_000, "10M ELEMENTS");
}

// ── Comparison sorts ─────────────────────────────────────────────────────────

/// Standard quicksort — median-of-three pivot, insertion sort for small slices
fn quicksort(data: &mut [f64]) {
    if data.len() <= 1 { return; }
    quicksort_inner(data);
}

fn quicksort_inner(data: &mut [f64]) {
    let n = data.len();
    if n <= 16 {
        // Insertion sort for small slices
        for i in 1..n {
            let key = data[i];
            let mut j = i;
            while j > 0 && data[j - 1] > key {
                data[j] = data[j - 1];
                j -= 1;
            }
            data[j] = key;
        }
        return;
    }

    // Median-of-three pivot selection
    let mid = n / 2;
    let last = n - 1;
    if data[0] > data[mid] { data.swap(0, mid); }
    if data[0] > data[last] { data.swap(0, last); }
    if data[mid] > data[last] { data.swap(mid, last); }
    // pivot is now at mid
    let pivot = data[mid];
    data.swap(mid, last - 1);

    // Three-way partition
    let mut lt = 0usize;
    let mut gt = last;
    let mut i = 0usize;
    while i < gt {
        if data[i] < pivot {
            data.swap(i, lt);
            lt += 1;
            i += 1;
        } else if data[i] > pivot {
            gt -= 1;
            data.swap(i, gt);
        } else {
            i += 1;
        }
    }

    quicksort_inner(&mut data[..lt]);
    quicksort_inner(&mut data[gt..]);
}

/// std::sort_unstable — Rust's built-in pdqsort (pattern-defeating quicksort)
/// This is the gold standard: introsort variant with adaptive behaviour
fn std_sort(data: &mut [f64]) {
    data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
}

/// Merge sort — classic O(n log n) stable sort
fn merge_sort(data: &mut Vec<f64>) {
    let n = data.len();
    if n <= 1 { return; }
    let mid = n / 2;
    let mut left = data[..mid].to_vec();
    let mut right = data[mid..].to_vec();
    merge_sort(&mut left);
    merge_sort(&mut right);
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] { data[k] = left[i]; i += 1; }
        else { data[k] = right[j]; j += 1; }
        k += 1;
    }
    while i < left.len() { data[k] = left[i]; i += 1; k += 1; }
    while j < right.len() { data[k] = right[j]; j += 1; k += 1; }
}

struct CompResult {
    input_type: String,
    n: usize,
    quicksort_us: u64,
    std_sort_us: u64,
    merge_sort_us: u64,
    visionsort_us: u64,
}

fn run_comparison(input_type: &str, original: &[f64]) -> CompResult {
    let n = original.len();

    // Each sort gets its own fresh copy
    let mut d1 = original.to_vec();
    let t = Instant::now();
    quicksort(&mut d1);
    let quicksort_us = t.elapsed().as_micros() as u64;

    let mut d2 = original.to_vec();
    let t = Instant::now();
    std_sort(&mut d2);
    let std_sort_us = t.elapsed().as_micros() as u64;

    let mut d3 = original.to_vec();
    let t = Instant::now();
    merge_sort(&mut d3);
    let merge_sort_us = t.elapsed().as_micros() as u64;

    // VisionSort — call the actual optimised library
    let mut d4 = original.to_vec();
    let t = Instant::now();
    vision_sort(&mut d4);
    let visionsort_us = t.elapsed().as_micros() as u64;

    // Verify all correct
    for (name, d) in [("quicksort",&d1),("std_sort",&d2),("merge_sort",&d3),("visionsort",&d4)] {
        if !d.windows(2).all(|w| w[0] <= w[1]) {
            eprintln!("SORT FAILED: {} on {}", name, input_type);
        }
    }

    CompResult { input_type: input_type.to_string(), n, quicksort_us, std_sort_us, merge_sort_us, visionsort_us }
}
