# MNIST Three.js Visualizer

Live demo: https://dgkim16.github.io/mnist_three

An interactive browser app that visualizes a TensorFlow.js MNIST CNN in Three.js, allows filter/weight inspection, and supports hand-drawn digit inference.

## Features
- 3D layer visualization with legend and inspect mode.
- Filter preview for Conv2D kernels.
- Dense edge rendering with sign/magnitude encoding.
- Hand-drawing panel with draw/erase, brush size, and top-3 prediction output.
- Keyboard shortcuts: `C` clear, `P` predict.
- Model loading fallback: attempts pre-trained model, then trains in browser if unavailable.

## Architecture
- `index.html`: responsive layout, UI containers, status panels.
- `script.js`: Three.js scene, interaction handlers, inference pipeline, evaluation.
- `data.js`: MNIST data loader and shuffled batching utilities.
- `docs/model-visualization.md`: assumptions and known visualization limitations.

## Setup
Run with a static server:

```bash
python3 -m http.server 8080
```

Open `http://localhost:8080`.

## Model artifacts
The repository includes model artifacts such as `my-model.json` and corresponding `.weights.bin` files, used as the first-choice load target for inference/visualization.

## Development workflow
See `CONTRIBUTING.md` for smoke tests, style rules, and PR checklist.
