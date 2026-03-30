# VisionSort

A prediction-driven adaptive sorting algorithm in Rust.

VisionSort sorts by building a probabilistic model of the input's value distribution and using that model to predict where each element belongs — falling back to comparisons only when predictions aren't confident enough. Each element placement refines the model, so the algorithm gets cheaper to run as it progresses.

On structured, low-entropy data (timestamps, prices, IDs, clustered values) it approaches O(n). On random data it falls back to O(n log n) — never worse than a standard sort.

---

## How It Works

The algorithm runs in five phases:

1. **Glance** — sample log₂(n) elements to build a coarse distribution model
2. **Segment** — walk the array once to detect and collect monotonic runs
3. **Disorder Map** — score each segment on disorder × entropy, build a priority heap
4. **Directed Fixation** — sort segments highest-priority first using one of four routes based on their scores
5. **Integrate** — k-way merge all sorted segments in O(n log k)

The four sort routes are:

| Disorder | Entropy | Route | Strategy |
|---|---|---|---|
| Low | Low | `NearlyFree` | Verify only — nearly free |
| Low | High | `Verify` | Confirm order, fix if needed |
| High | Low | `PlacementSort` | Predict position, place without comparison |
| High | High | `FullSort` | Introsort fallback — guaranteed O(n log n) |

`PlacementSort` is the novel route. It uses the distribution model to predict each element's output position, places it tentatively without comparison, then sorts only the small minority of elements where prediction confidence was too low or slots collided.

---

## Requirements

- Rust 1.65 or later (`cargo` and `rustc`)
- No external dependencies at runtime

Check your version:
```bash
rustc --version
cargo --version
```

If you don't have Rust, install it from [rustup.rs](https://rustup.rs).

---

## Installation

Clone or download the project, then build:

```bash
git clone <your-repo-url>
cd visionsort
cargo build
```

To use as a library in your own project, add to your `Cargo.toml`:

```toml
[dependencies]
visionsort = { path = "../visionsort" }
```

---

## Usage

### Simple function call

```rust
use visionsort::vision_sort;

fn main() {
    let mut data = vec![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    vision_sort(&mut data);
    println!("{:?}", data); // [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]
}
```

### Struct-based API (access the model after sorting)

```rust
use visionsort::VisionSort;

fn main() {
    let mut data = vec![5.0_f64, 2.0, 8.0, 1.0, 9.0, 3.0];
    
    let mut sorter = VisionSort::new();
    sorter.sort(&mut data);
    
    println!("{:?}", data);
    
    // Inspect the distribution model after sorting
    println!("Anchors: {:?}", sorter.model.anchors);
    println!("Final entropy: {}", sorter.model.entropy);
    println!("Observations: {}", sorter.model.observations);
}
```

### Sorting integers

VisionSort works on any type that implements `PartialOrd + Into<f64> + Copy`. The standard numeric types (`f32`, `f64`, `i32`, `i64`, `u32`, `u64`, etc.) all work out of the box:

```rust
use visionsort::vision_sort;

fn main() {
    let mut ints: Vec<f64> = vec![42.0, 7.0, 99.0, 1.0, 23.0];
    vision_sort(&mut ints);
    println!("{:?}", ints);
}
```

For integer types, cast to f64 first:

```rust
let mut raw: Vec<i64> = vec![100, 3, 57, 2, 99];
let mut data: Vec<f64> = raw.iter().map(|&x| x as f64).collect();
vision_sort(&mut data);
// map back if needed
```

### Sorting structured data

For structs, implement `Into<f64>` on the key you want to sort by:

```rust
use visionsort::vision_sort;

#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct Record {
    timestamp: f64,
    value: f64,
}

impl Into<f64> for Record {
    fn into(self) -> f64 {
        self.timestamp // sort by timestamp
    }
}

fn main() {
    let mut records = vec![
        Record { timestamp: 1700.0, value: 42.0 },
        Record { timestamp: 1500.0, value: 11.0 },
        Record { timestamp: 1600.0, value: 77.0 },
    ];
    vision_sort(&mut records);
}
```

---

## Running Tests

```bash
cargo test
```

Expected output: 16 tests, all passing.

```
running 16 tests
test tests::cost_curve_entropy_decreases ... ok
test tests::stress_clustered_10k ... ok
test tests::stress_duplicates_10k ... ok
test tests::stress_nearly_sorted_10k ... ok
test tests::stress_random_1k ... ok
test tests::stress_random_10k ... ok
test tests::stress_reversed_10k ... ok
test tests::test_all_same ... ok
test tests::test_already_sorted ... ok
test tests::test_empty ... ok
test tests::test_nearly_sorted ... ok
test tests::test_random_small ... ok
test tests::test_reverse_sorted ... ok
test tests::test_route_decision ... ok
test tests::test_single ... ok
test tests::test_two_elements ... ok

test result: ok. 16 passed; 0 failed
```

To run a specific test:

```bash
cargo test stress_random_10k
cargo test cost_curve
```

To run tests with output visible:

```bash
cargo test -- --nocapture
```

---

## Project Structure

```
visionsort/
├── Cargo.toml          — package manifest
├── Cargo.lock          — dependency lockfile
├── README.md           — this file
├── WHITEPAPER.md       — full algorithm design and complexity analysis
└── src/
    └── lib.rs          — full implementation + tests
```

**Key types in `lib.rs`:**

- `vision_sort<T>(data: &mut [T])` — top-level public function
- `VisionSort<T>` — struct with persistent model, use when you want to inspect the model post-sort
- `DistributionModel` — the learning core: anchors, entropy, predict(), update()
- `Segment` — a scored monotonic run with a route assignment
- `SortRoute` — the four-quadrant routing enum

---

## Constraints

**Type requirements:** `T: PartialOrd + Into<f64> + Copy`

The `Into<f64>` bound is required for the distribution model, which operates in f64 to interpolate predicted positions. If you're sorting a type that can't be meaningfully mapped to f64, VisionSort is not the right tool.

**Stability:** VisionSort is not a stable sort. Equal elements may appear in any order in the output.

**Memory:** Phase 5 (k-way merge) snapshots each sorted segment into a separate Vec before merging. Peak memory usage is approximately 2× the input size.

**Numeric edge cases:** NaN values in float inputs will produce unspecified behavior. Filter NaNs before sorting.

---

## Whitepaper

For full design rationale, complexity analysis, and relationship to existing adaptive sorts, see [WHITEPAPER.md](./WHITEPAPER.md).

---

## License

MIT — do whatever you want with it. Credit appreciated but not required.
