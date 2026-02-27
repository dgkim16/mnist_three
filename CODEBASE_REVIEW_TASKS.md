# Codebase Review: Issues and Improvement Tasks

This review identifies concrete follow-up work in four categories: **correctness**, **visual quality**, **interactivity**, and **documentation**.

## 1) Correctness

### C1. Fix input preprocessing mismatch for hand-drawn digit prediction
- **Issue**: The drawing canvas uses white-on-black and direct grayscale conversion without inverting, centering, or normalizing exactly like training data, which can degrade prediction quality.
- **Impact**: Incorrect predictions on user-drawn digits.
- **Tasks**:
  1. Add a preprocessing pipeline that converts canvas pixels to `[0,1]`, inverts if needed, and matches MNIST conventions.
  2. Center drawn content using bounding-box extraction and optional padding before resize.
  3. Dispose temporary tensors created during prediction (`tensor`, `prediction`, `argMax` result) to avoid memory leaks.
  4. Add a deterministic unit/integration check for preprocessing output shape and range.

### C2. Correct dense/flatten weight indexing in edge visualization
- **Issue**: In `drawWeights`, the dense-path slicing uses hardcoded `784` and assumes a specific layer ordering, which can produce wrong edge mapping when architecture changes.
- **Impact**: Misleading visualization and incorrect weight lines.
- **Tasks**:
  1. Derive slice sizes from actual tensor shapes (`modelWeights[n].shape`) instead of constants.
  2. Build a layer-name-to-weight map so indexing is explicit (e.g., `dense_Dense/kernel`).
  3. Add guards for out-of-range neuron indices and unknown layer types.

### C3. Handle normalization edge case for zero-variance filters
- **Issue**: `normalizeFilter` divides by `(max - min)` without guarding against `0`.
- **Impact**: `NaN` grayscale values for constant filters.
- **Tasks**:
  1. Add safe branch for `max === min` (e.g., return mid-gray or zeros).
  2. Add a tiny test helper function that validates expected normalization behavior.

### C4. Fix resource lifecycle and memory usage around evaluation
- **Issue**: `doPrediction` returns `preds` and `labels`, but only `labels.dispose()` is called in some consumers.
- **Impact**: Tensor leaks across repeated evaluation runs.
- **Tasks**:
  1. Ensure both `preds` and `labels` are disposed in `showAccuracy` and `showConfusion`.
  2. Wrap intermediate prediction/evaluation steps in `tf.tidy` where possible.
  3. Add periodic `tf.memory()` logging behind a debug flag.

### C5. Improve model-loading fallback behavior
- **Issue**: Runtime assumes `./my-model.json` exists and loads successfully.
- **Impact**: App fails silently or hard-fails when the model artifact is missing/corrupt.
- **Tasks**:
  1. Add explicit error UI for model load failure.
  2. Provide fallback path: train a model if loading fails (or prompt to download).
  3. Add a “model status” badge in UI.

---

## 2) Visual improvements

### V1. Replace rigid fixed-size layout with responsive UI
- **Issue**: Renderer and camera use hardcoded `800x800`, and page elements are appended with minimal structure.
- **Impact**: Poor layout on small/large screens and mixed controls alignment.
- **Tasks**:
  1. Use CSS layout containers (`left: 3D view`, `right: controls/panels`) with responsive breakpoints.
  2. Update renderer and camera on window resize.
  3. Add spacing/typography tokens for visual consistency.

### V2. Improve neuron/link visual encoding and legend
- **Issue**: Layer colors are not explained; weight edges only use red/green sign and no magnitude scaling.
- **Impact**: Visualization is hard to interpret.
- **Tasks**:
  1. Add on-screen legend for layer colors and edge semantics.
  2. Encode edge magnitude using opacity/width with clamped scaling.
  3. Add hover tooltip with layer/neuron metadata.

### V3. Improve filter rendering readability
- **Issue**: Filter previews are minimal and unstructured.
- **Impact**: Hard to compare filters and understand what’s shown.
- **Tasks**:
  1. Render filter cards with title, min/max values, and consistent sizing.
  2. Use a contrast-aware colormap (or explicit grayscale scale marker).
  3. Add “pin filter” and multi-filter grid view for comparisons.

---

## 3) Interactivity improvements

### I1. Replace click-toggle hover mode with explicit interaction controls
- **Issue**: A global click toggles `canHover`, which is opaque and easy to trigger accidentally.
- **Impact**: Confusing UX.
- **Tasks**:
  1. Replace with explicit toggle button (`Inspect mode: on/off`).
  2. Display current mode in the UI.
  3. Disable inspect mode while drawing on canvas.

### I2. Add camera controls and reset behavior
- **Issue**: OrbitControls is imported but not enabled.
- **Impact**: Limited exploration of 3D scene.
- **Tasks**:
  1. Enable `OrbitControls` with damping.
  2. Add “Reset Camera” button.
  3. Persist camera target when re-rendering scene nodes.

### I3. Improve drawing experience and prediction feedback
- **Issue**: Prediction is an alert only; no confidence display.
- **Impact**: Low trust and poor iterative workflow.
- **Tasks**:
  1. Show top-3 class probabilities in a panel.
  2. Add brush size and erase mode.
  3. Add keyboard shortcuts (`C` clear, `P` predict).

### I4. Add loading and long-operation feedback
- **Issue**: Dataset/model loading and eval have no explicit loading state.
- **Impact**: Appears unresponsive.
- **Tasks**:
  1. Add spinner/status text for `load data`, `load model`, and `evaluate`.
  2. Disable buttons during active operations.
  3. Surface error messages inline instead of `alert`.

---

## 4) Documentation improvements

### D1. Expand README from single URL to full project guide
- **Issue**: README currently only contains a link.
- **Impact**: New contributors cannot run, debug, or extend the project quickly.
- **Tasks**:
  1. Add project overview and architecture diagram section.
  2. Document setup/run instructions (static server, expected browser support).
  3. Document model artifacts (`my-model*.json/.bin`) and how they were produced.

### D2. Add developer notes for model/visualization assumptions
- **Issue**: Key assumptions are implicit in code comments.
- **Impact**: Changes can silently break visualization logic.
- **Tasks**:
  1. Add `docs/model-visualization.md` covering layer mapping, neuron semantics, and known limitations.
  2. Document tensor shapes used in each stage (input, conv, flatten, dense).
  3. Add “known issues” section with planned fixes.

### D3. Add contribution/testing guidance
- **Issue**: No documented quality gates.
- **Impact**: Inconsistent changes and regressions.
- **Tasks**:
  1. Add minimal smoke-test checklist (load app, inspect nodes, draw/predict).
  2. Define formatting/linting preference for JS and HTML.
  3. Add PR template/checklist for correctness + UX checks.

---

## Suggested execution order (high impact first)
1. **C1 + C4** (prediction correctness + memory safety)
2. **C2 + C3** (weight/filter correctness)
3. **V1 + I1 + I2** (core UX and navigation)
4. **I3 + I4 + V2** (clarity and confidence)
5. **D1 + D2 + D3 + V3** (maintainability and polish)
