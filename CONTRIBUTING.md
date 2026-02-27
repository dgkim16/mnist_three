# Contributing

## Local run
Use any static file server, for example:

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

## Smoke test checklist
1. App loads and status changes to **Ready**.
2. Model status shows loaded model or training fallback.
3. Inspect mode can be toggled and hovering neurons updates lines/filter panel.
4. Drawing canvas works with draw/erase and brush-size slider.
5. Predict shows top-3 probabilities.
6. `C` clears and `P` predicts.

## Style guidance
- Keep JS modular and avoid magic numbers.
- Dispose temporary tensors (`tf.tidy` + explicit `.dispose()`).
- Prefer responsive layout; avoid fixed-size viewport assumptions.

## Pull request checklist
- [ ] Correctness updates include tensor lifecycle review.
- [ ] Visual/interaction changes tested at desktop and narrow width.
- [ ] Any model-shape assumptions documented in `docs/model-visualization.md`.
