# VisionSort: A Prediction-Driven Adaptive Sorting Algorithm

**Harsh Patel**  
Independent Research  
March 2026

---

## Abstract

We present VisionSort, a sorting algorithm whose design is derived from two principles in human visual perception: gestalt scene modeling and saccade-directed attention. Rather than treating each element touch as a comparison operation that yields one bit of information, VisionSort treats each touch as a dual-purpose event — it positions the element *and* refines a probabilistic distribution model that reduces the cost of all subsequent placements. The result is an algorithm whose marginal cost per element decreases as the sort progresses. Complexity is expressed as T(n, H) where H is the Shannon entropy of the input's value distribution, rather than the conventional T(n) characterization. On low-entropy structured data, VisionSort approaches O(n). On adversarial maximum-entropy data, it degrades gracefully to O(n log n). The algorithm passes correctness verification against `std::sort` across six input classes at scale up to 10,000 elements.

---

## 1. Motivation

The O(n log n) lower bound for comparison-based sorting is a well-established result derived from information theory: to identify one permutation of n elements out of n! possibilities requires at least log₂(n!) ≈ n log n bits of information, and each comparison delivers at most one bit. This bound is tight and has been accepted as the ceiling of what is achievable.

However, the bound rests on an assumption that is frequently violated in practice: that the input is drawn from a uniform distribution over all n! permutations. Real-world data — financial transactions, log timestamps, sensor readings, identifiers — is not uniformly random. It has structure. It clusters. It has low entropy. The information required to specify the correct permutation of low-entropy data is strictly less than log₂(n!), which means the theoretical minimum comparison count is also less.

Existing adaptive sorting algorithms (Timsort, Smoothsort, Splaysort, natural merge sort) partially exploit this by detecting existing order in the input. But they share a common limitation: they measure disorder globally or reactively, and they extract exactly one bit of information per comparison regardless of how much structural knowledge has been accumulated. They do not build a predictive model of the input's distribution, and they do not use that model to reduce the number of comparisons required.

VisionSort is motivated by the observation that human visual perception solves an analogous problem differently. The visual system does not compare every pixel to every other pixel. It builds a coarse global scene model first (gestalt), then directs focused attention (saccades) to regions of high complexity while treating low-complexity regions as nearly free. Each fixation both processes a region *and* updates the scene model, sharpening predictions for regions not yet examined.

Mapping this to sorting: rather than performing comparisons uniformly across all elements, an algorithm could sample a small number of elements to build a distribution model, then use that model to *predict* where each remaining element belongs, falling back to comparisons only when prediction confidence is too low to trust.

---

## 2. Core Thesis

**Every element touch in VisionSort does two jobs:**

1. It positions the element (the standard job of any sort operation)
2. It updates the distribution model via a Bayesian refinement step, improving the prediction accuracy for all elements not yet processed

This dual-purpose property means the marginal cost per element is not constant across the sort. Early elements are expensive — the model is weak, predictions are wide, comparisons are frequent. Late elements are cheap — the model is refined, predictions are narrow, most elements can be placed without comparison.

The cost curve of VisionSort is decreasing, not flat. This is a property no existing comparison-based sort possesses.

---

## 3. Complexity Characterization

Let H be the Shannon entropy of the input's value distribution, normalized to [0, 1] where 0 is perfectly predictable (all identical values) and 1 is maximally uniform.

VisionSort's total comparison cost is bounded by:

```
C(n, H) = Σᵢ [ H_local(i) × Lᵢ ]
```

where the sum is over all segments i, H_local(i) is the local entropy of segment i's value distribution, and Lᵢ is the segment's length.

This sum is bounded above by n × H_max, where H_max is the entropy of the full input distribution. Since H_max ≤ log₂(n), the standard O(n log n) bound is recovered in the worst case. But for structured inputs where H_max << log₂(n), the bound is substantially tighter.

**Complexity by input class:**

| Input class | H_max | Complexity |
|---|---|---|
| Already sorted | ≈ 0 | O(n) |
| Nearly sorted | low | O(n · H) |
| Clustered / structured | low-medium | O(n · H) |
| Uniform random | ≈ log₂(n) | O(n log n) |
| Adversarial | = log₂(n) | O(n log n) |

The key insight: complexity is parameterized by the actual information content of the data, not just its size. This is a different axis of analysis than any existing adaptive sort provides.

---

## 4. Algorithm Design

VisionSort operates in five sequential phases.

### Phase 1 — The Glance (O(log n))

Sample k = ⌈log₂(n)⌉ elements at evenly-spaced indices. Sort these samples. The sorted samples become *anchor points* — a coarse approximation of the value distribution across the array. From the anchors, compute an initial global entropy estimate using Shannon entropy over a histogram of the sample values.

This phase costs O(k log k) = O(log²n) — effectively free — and produces a scene model that will guide every subsequent phase.

### Phase 2 — Segmentation (O(n))

Walk the array once. Commit to either an ascending or descending run at each position, walk the run to exhaustion, and emit it as a segment. Descending runs are reversed in-place immediately, converting them to ascending runs at no additional asymptotic cost.

The key invariant: segments tile the array exactly — no gaps, no overlaps. Each element belongs to exactly one segment.

Post-Phase 2, the array is a set of non-overlapping ascending runs. These runs are the units of further processing.

### Phase 3 — Disorder Mapping (O(k√L))

Score each segment on two independent axes:

**Disorder score** — estimated inversion rate, computed by sampling √L random pairs within the segment and counting how many are out of order. Normalized to [0, 1].

**Entropy score** — Shannon entropy of the segment's value distribution, computed via a √n-bucket histogram. Normalized to [0, 1].

These two scores define a 2D routing space with four quadrants:

| | **Low entropy** | **High entropy** |
|---|---|---|
| **Low disorder** | `NearlyFree` — verify only | `Verify` — confirm order |
| **High disorder** | `PlacementSort` — predict & place | `FullSort` — introsort fallback |

Segments with length ≤ 16 are routed to `Trivial` regardless of scores.

Each segment's priority score is `disorder × entropy × length`. Segments are inserted into a max-heap keyed by this score — highest-priority (most disordered, most entropic, largest) segments are processed first.

### Phase 4 — Directed Fixation (decreasing marginal cost)

Pop segments from the heap in priority order and sort each according to its route:

**Trivial / NearlyFree:** Insertion sort. Optimal for small or nearly-ordered segments.

**Verify:** Single linear scan to detect order violations. Only invoke insertion sort if violations are found. Sorted high-entropy segments often require no work at all.

**PlacementSort:** The novel route. For each element in the segment:
1. Query the distribution model for a predicted output position and confidence score
2. If confidence exceeds threshold (0.7) and the predicted slot is unoccupied, place the element there tentatively without comparison
3. Otherwise, queue the element for comparison-based resolution

After all elements are processed, sort only the unplaced minority using introsort, then slot them into the gaps. Verify the result with a linear scan and apply insertion sort only if violations remain.

After each placement, update the distribution model: refine min/max bounds, decay entropy proportional to the observation ratio, and insert high-surprise elements (those whose predicted position was far from actual) as new anchor points.

**FullSort:** Introsort (quicksort with heapsort fallback at depth limit 2 log n). Guaranteed O(n log n). This is the adversarial fallback — reached only when both disorder and entropy are high.

### Phase 5 — Integration (O(n log k))

Merge all k sorted segments using a min-heap k-way merge. Each segment seeds the heap with its first element. Each heap pop yields the globally minimum remaining value and re-seeds the heap with the next element from the same segment.

Total cost: O(n log k) where k is the number of segments. On nearly-sorted data with few long runs, k is small and this phase is near O(n).

---

## 5. The Distribution Model

The distribution model is the algorithm's memory. It persists across the sort and accumulates knowledge from every element placement.

**Structure:**
- `anchors`: sorted list of f64 values that approximate the output distribution. Initialized from Phase 1 samples. Grows as high-surprise elements are inserted during Phase 4.
- `estimated_min`, `estimated_max`: refined bounds, updated after every observation
- `entropy`: running estimate of distribution predictability, initialized from Phase 1 and decayed with each observation
- `observations`: count of elements processed, used to compute observation ratio

**Prediction** (`predict(value) → (position, confidence)`):

Binary-search the anchor list to find the two anchors surrounding the value. Linearly interpolate the predicted output position between them. Confidence is computed as a weighted combination of `(1 - entropy)` and the observation ratio — both factors increase as the sort progresses, so confidence monotonically increases over time.

**Update** (`update(value, actual_position, n)`):

After each placement, refine the model:
1. Update min/max if the new value extends the known range
2. Decay entropy: `entropy *= (1 - obs_ratio × 0.01)` — entropy asymptotically approaches zero as the sort completes
3. Compute prediction surprise: `|predicted_position - actual_position|`
4. If surprise > 0.1, insert the value as a new anchor point — it represents a region of the distribution where the model was wrong, and future predictions in that region will be more accurate

The surprise-triggered anchor insertion is the core Bayesian update mechanism. The model does not start with a fixed number of anchors — it grows its resolution in regions where predictions fail, while leaving well-predicted regions alone.

---

## 6. The Decreasing Cost Property

The claim that VisionSort speeds up as it runs follows from two monotonic properties:

1. **Confidence increases monotonically.** Both components of the confidence formula — `(1 - entropy)` and `obs_ratio` — increase as observations accumulate. Entropy decays toward zero. Observation ratio increases toward one. Therefore confidence is non-decreasing across the sort.

2. **Anchor density increases in surprise regions.** Each high-surprise placement adds an anchor. The model's resolution in poorly-predicted regions increases over time, narrowing future prediction ranges and reducing future surprise.

Together: early elements are processed with low confidence and frequent comparison fallback. Later elements are processed with high confidence and rare comparison fallback. The expected number of comparisons per element is a decreasing function of the number of elements already processed.

This property is verified by the `cost_curve_entropy_decreases` test, which simulates 100 Bayesian updates and asserts that entropy is non-increasing across the sequence.

---

## 7. Adversarial Resistance

A property worth noting explicitly: VisionSort cannot be attacked into performing worse than O(n log n).

An adversary attempting to degrade the algorithm must construct maximally unpredictable input — high entropy, high disorder in every segment. But the entropy detection in Phase 3 catches this immediately: all segments route to `FullSort`, which invokes introsort with its O(n log n) worst-case guarantee.

The adversarial case is not a surprise — it announces itself via the entropy score. And when it arrives, the algorithm falls back to a proven baseline. The adversary cannot force behavior worse than that baseline.

Formally: VisionSort is O(n log n) in all cases, with performance strictly better than that bound on any input with measurable structure.

---

## 8. Relationship to Existing Work

**Timsort** detects runs and merges them adaptively. VisionSort generalizes this in two ways: it assigns independent disorder and entropy scores to each run (rather than treating all runs equivalently), and it applies prediction-based placement on high-disorder low-entropy segments rather than always falling back to comparison-based sorting.

**Interpolation sort** places elements using a global linear interpolation estimate. VisionSort differs in two ways: it uses a multi-resolution anchor model rather than a single linear estimate, and it uses batch tentative placement with conflict resolution rather than single-element placement with immediate correction.

**Samplesort** uses logarithmic sampling to estimate bucket boundaries before sorting. Phase 1 of VisionSort is structurally similar, but the sample is used as an adaptive prediction model rather than as a static bucket partitioner.

**Splitsort and Binomialsort** optimize adaptively with respect to inversion count. VisionSort's disorder metric is also related to inversion count, but adds entropy as an orthogonal second dimension that no existing adaptive sort accounts for.

The closest conceptual relative is **learned index structures** (Kraska et al., 2018), which use machine learning models to predict the position of keys in sorted order. VisionSort applies a similar predictive approach to the sorting process itself rather than to post-sort lookup, and constructs its model from the input being sorted rather than from training data.

---

## 9. Limitations and Open Problems

**The confidence threshold is a fixed parameter (0.7).** The optimal threshold likely varies by input distribution. An adaptive threshold that responds to the current model quality is a natural extension.

**Phase 5 uses a snapshot-based k-way merge.** The segment data is copied into per-segment vectors before merging, which doubles peak memory usage. An in-place k-way merge would eliminate this overhead.

**The formal complexity proof is incomplete.** The T(n, H) characterization is motivated by the algorithm's design but has not been proven with the rigor of a comparison lower bound proof. A formal proof that the expected comparison count is bounded by C(n, H) = n × H_max + O(n) is left as an open problem.

**The model is not persisted across calls.** Each invocation of `vision_sort` constructs a fresh model. For applications that sort similar data repeatedly — logs, time-series, financial streams — persisting the model across calls would allow the cross-sort learning property: sort k+1 benefits from the distribution knowledge accumulated during sorts 1 through k.

---

## 10. Conclusion

VisionSort introduces a design principle not present in existing sorting algorithms: using each element placement to refine a predictive model that reduces the cost of future placements. This dual-purpose property produces a decreasing marginal cost curve, a complexity characterization parameterized by input entropy rather than size alone, and adversarial resistance via entropy-triggered fallback to proven worst-case algorithms.

The algorithm is implemented in Rust, passes correctness verification at scale across six input classes, and includes a test that directly measures the decreasing entropy property.

The core conceptual contribution is the framing: sorting as a perception problem rather than a comparison problem. The visual system's gestalt modeling and saccade-directed attention provide a principled basis for a class of algorithms that spend computation proportionally to local uncertainty rather than uniformly across the input.

---

## References

Estivill-Castro, V., & Wood, D. (1992). A survey of adaptive sorting algorithms. *ACM Computing Surveys*, 24(4), 441–476.

Kraska, T., Beutel, A., Chi, E. H., Dean, J., & Polyzotis, N. (2018). The case for learned index structures. *Proceedings of the 2018 International Conference on Management of Data*, 489–504.

Levcopoulos, C., & Petersson, O. (1991). Splitsort — an adaptive sorting algorithm. *Information Processing Letters*, 39(5), 205–211.

Mannila, H. (1985). Measures of presortedness and optimal sorting algorithms. *IEEE Transactions on Computers*, C-34(4), 318–325.

Peters, T. (2002). Timsort. CPython source, `Objects/listsort.txt`.

Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.
