/* tslint:disable */
/* eslint-disable */

export class WasmModel {
    free(): void;
    [Symbol.dispose](): void;
    anchor_count(): number;
    entropy(): number;
    estimated_max(): number;
    estimated_min(): number;
    constructor();
    observations(): number;
    predict(value: number): number;
    predict_confidence(value: number): number;
    /**
     * Seed the model with initial samples (like Phase 1 — The Glance)
     */
    seed(samples: Float64Array): void;
    update(value: number, pos: number, n: number): void;
}

export function wasm_traced_sort(data: Float64Array): any;

export function wasm_vision_sort(data: Float64Array): Float64Array;

export function wasm_vision_sort_with_trace(data: Float64Array): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmmodel_free: (a: number, b: number) => void;
    readonly wasm_traced_sort: (a: number, b: number) => any;
    readonly wasm_vision_sort: (a: number, b: number) => [number, number];
    readonly wasm_vision_sort_with_trace: (a: number, b: number) => any;
    readonly wasmmodel_anchor_count: (a: number) => number;
    readonly wasmmodel_entropy: (a: number) => number;
    readonly wasmmodel_estimated_max: (a: number) => number;
    readonly wasmmodel_estimated_min: (a: number) => number;
    readonly wasmmodel_new: () => number;
    readonly wasmmodel_observations: (a: number) => number;
    readonly wasmmodel_predict: (a: number, b: number) => number;
    readonly wasmmodel_predict_confidence: (a: number, b: number) => number;
    readonly wasmmodel_seed: (a: number, b: number, c: number) => void;
    readonly wasmmodel_update: (a: number, b: number, c: number, d: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
