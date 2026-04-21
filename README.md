# MNIST CNN — draw & inspect

Live: https://dgkim16.github.io/mnist_three

Browser-only MNIST digit recogniser built on TensorFlow.js, with a Three.js visualisation that shows every convolutional layer's activations and the learned kernels themselves as three-dimensional voxel volumes.

## Model

`Conv2D(8 filters, 5×5, ReLU) → MaxPool 2×2 → Conv2D(16 filters, 5×5, ReLU) → Flatten → Dense(10, softmax)`

Pretrained weights are loaded from `my-model.json`. `getModel()` and `train()` remain in `script.js` for retraining but are not on the runtime path.

## Running

No build step. ES-module imports break over `file://`, so serve the directory over HTTP:

```
python -m http.server 8000
```

Then open `http://localhost:8000/`.

## Inference fixes

The first version trained correctly but mispredicted almost every user-drawn digit. Five fixes, in order of impact:

1. **Normalisation.** `.div(255.0)` was missing before `model.predict`. A 0–255 input pushed every conv activation roughly 250× past anything seen during training, so the softmax outputs were essentially noise.

2. **Bilinear replacing nearest-neighbour.** With a stroke radius of 10 px and a 10:1 ratio, nearest-neighbour sampled a single pixel out of each 10×10 tile and frequently missed the stroke entirely, producing a sparse dotted glyph. `resizeBilinear([28, 28])` — or, more precisely, an `avgPool` with `[10, 10]` window and stride 10, which is the exact box filter for this ratio — was the correct operator.

3. **Match MNIST framing.** Real MNIST digits occupy a ~20×20 region centred by pixel mass inside a 28×28 image with a 4-px black margin. A user-drawn glyph that filled the 280×280 canvas was being resized into a 28×28 glyph that filled the entire frame, which is out of distribution for this CNN. Preprocessing now finds the ink bounding box, scales its longest side to 20, and pastes it centred into a 28×28 black frame.

4. **Stroke width.** With `arc(..., 10, ...)` an MNIST-scaled stroke is roughly 2 px wide, while training strokes are 3–4 px wide and anti-aliased. `lineWidth = 20` on the draw context, combined with bilinear downsampling for AA, matches the training distribution.

5. **Tensor disposal.** Wrapping the forward pass in `tf.tidy(() => { ... })` and explicitly disposing intermediate tensors is orthogonal to correctness but prevents a tensor leak on every click.

## 3D visualisation

An earlier version rendered each activation channel as a `PlaneGeometry` textured with a `CanvasTexture`. Every plane was parallel to the XY plane, so the Z axis carried no information — rotating the camera never revealed a new spatial relationship. It was 2D content transported into a Three.js scene, not a 3D visualisation.

The current version uses Three.js for three things that require genuine spatial depth:

### Voxel volumes per layer

Every feature map is rendered as an `InstancedMesh` of small cubes, one instance per `(channel, row, col)` cell, with axes `Y = row`, `Z = col`, `X = layer origin + channel offset`. Each voxel's colour is the per-channel-normalised activation at that cell, set via `setColorAt` inside `updateFeatureMaps`. `MeshLambertMaterial` combined with one ambient and two directional lights shades box faces differently so the block reads as a solid volume rather than a silhouette.

Because row, column, and channel each occupy a distinct world axis, orbiting the camera exposes a new projection of the same data rather than re-showing the same flat collage.

### Receptive-field overlays on hover

The raycaster picks the instance under the cursor; on any change the overlay group is rebuilt:

- **Conv1**: 25 line segments back to the matching 5×5 input patch.
- **Pool1**: 4 line segments to the same-channel 2×2 conv1 window.
- **Conv2**: a single wireframe hull box over the 5×5 spatial region spanning all 8 pool1 channels. Drawing the real 200 line segments is visual noise with no added information; a hull communicates "this voxel depends on this 5×5×8 region" directly.

The line-segment math reflects each layer's actual receptive field — valid-padding `Conv2D(5×5)` shifts coordinates by `+0..+4`, `MaxPool(2×2)` shifts by `2y + 0..1`.

### Kernel inspector on hover

The learned convolution kernel for the hovered filter is materialised as a floating voxel volume above the layer: orange for positive weights, blue for negative, opacity proportional to `|w| / max|w|`. For Conv1 this is a 5×5 slab over a single input channel. For Conv2 this is a 5×5×8 volume — an inherently three-dimensional object that cannot be faithfully projected to a plane without either hiding an axis or tiling channels into an arbitrary 2D grid.

Kernel weights are cached once by `extractConvKernels()` at load time; hover does not re-read WebGL buffers.

### Overlay lifetime

`pointermove` fires continuously. `clearHoverVisuals()` runs before every rebuild and disposes each overlay child's geometry and material explicitly — neglecting this leaks GPU memory on every frame the mouse is in motion.

Flatten and Dense are deliberately excluded from the 3D scene: they discard spatial structure, and the prediction bar chart already covers the softmax output.

## Tensor hygiene

Every stroke triggers a forward pass, so leaks accumulate quickly.

- `pre` (the preprocessed input tensor) and every tensor in the `activations` array from `activationModel.predict` are explicitly disposed after `updateFeatureMaps` returns.
- `updateFeatureMaps` builds each normalised per-layer tensor inside `tf.tidy` and disposes it immediately after `data()` has pulled values to JS.
- The GPU→CPU read happens once per layer rather than once per channel; the 2D channel canvases are then written via `putImageData` from the same JS buffer, avoiding a second read.
