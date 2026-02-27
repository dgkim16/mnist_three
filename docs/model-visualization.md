# Model Visualization Notes

## Layer mapping
The visualizer maps Keras layer names to these display types:
- `conv2d*` -> Conv2D (one sphere per filter)
- `max_pooling2d*` -> MaxPooling2D (capped preview count)
- `flatten*` -> Flatten (capped preview count)
- `dense*` -> Dense (one sphere per output unit)

## Tensor shape assumptions
- Input drawing is converted to `Tensor[1, 28, 28, 1]`.
- Conv kernels are read as `[kernelH, kernelW, inChannels, outChannels]`.
- Dense kernel is read as `[inputUnits, outputUnits]`.

## Edge rendering semantics
- Edge color: green for positive weight, red for negative weight.
- Edge opacity: scales with absolute magnitude and is clamped.

## Known limitations
- Conv and pooling layers are spatial operators; displayed “neurons” are conceptual, not direct one-to-one activations.
- Flatten/MaxPool previews are capped to keep scene size manageable.
- Dense edges currently show sign+magnitude, not activation flow for a specific sample.
