import * as T from "https://unpkg.com/three@v0.149.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@v0.149.0/examples/jsm/controls/OrbitControls.js";
import { MnistData } from './data.js';

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];
const DEBUG_MEMORY = false;
const MAX_PREVIEW_NODES = 64;

const ui = {
  status: document.getElementById('status'),
  modelStatus: document.getElementById('modelStatus'),
  buttonRow: document.getElementById('buttonRow'),
  drawTools: document.getElementById('drawTools'),
  drawCanvas: document.getElementById('drawCanvas'),
  resizedCanvas: document.getElementById('resizedCanvas'),
  predictionText: document.getElementById('predictionText'),
  filtersContainer: document.getElementById('filtersContainer'),
  sceneCanvasWrap: document.getElementById('sceneCanvasWrap'),
  topInputPreview: document.getElementById('topInputPreview')
};

const raycaster = new T.Raycaster();
const mouse = new T.Vector2();
let inspectMode = false;

let scene;
let camera;
let renderer;
let controls;
let nodes;
let model;
let modelWeights;
let layerNodeMap = [];
let drawing = false;
let eraseMode = false;
let brushSize = 10;

const flattenMat = new T.MeshBasicMaterial({ color: 0x0000ff });
const denseMat = new T.MeshBasicMaterial({ color: 0xff4500 });
const conv2dMat = new T.MeshBasicMaterial({ color: 0x00ff00 });
const maxPoolingMat = new T.MeshBasicMaterial({ color: 0xff00ff });

document.addEventListener('DOMContentLoaded', run);

function setStatus(message) {
  ui.status.textContent = message;
}

function setBusyState(busy) {
  document.querySelectorAll('button').forEach(btn => { btn.disabled = busy; });
}

function updateModelStatus(message) {
  ui.modelStatus.textContent = message;
}

function getModel() {
  const nextModel = tf.sequential();
  nextModel.add(tf.layers.conv2d({
    inputShape: [28, 28, 1], kernelSize: 5, filters: 8, strides: 1,
    activation: 'relu', kernelInitializer: 'varianceScaling'
  }));
  nextModel.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
  nextModel.add(tf.layers.conv2d({
    kernelSize: 5, filters: 16, strides: 1,
    activation: 'relu', kernelInitializer: 'varianceScaling'
  }));
  nextModel.add(tf.layers.flatten());
  nextModel.add(tf.layers.dense({ units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax' }));
  nextModel.compile({ optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return nextModel;
}

async function train(nextModel, data) {
  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });
  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  try {
    await nextModel.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 5,
      shuffle: true
    });
  } finally {
    trainXs.dispose();
    trainYs.dispose();
    testXs.dispose();
    testYs.dispose();
  }
}

async function extractWeights(nextModel) {
  const map = new Map();
  nextModel.layers.forEach(layer => {
    const layerWeights = layer.getWeights();
    if (!layerWeights.length) return;
    const kernel = layerWeights[0];
    map.set(layer.name, {
      name: layer.name,
      shape: kernel.shape,
      isConv: layer.name.includes('conv2d'),
      weights: layer.name.includes('conv2d') ? kernel.arraySync() : Array.from(kernel.dataSync())
    });
  });
  return map;
}

function getLayerType(layerName) {
  if (layerName.includes('conv2d')) return 'Conv2D';
  if (layerName.includes('dense')) return 'Dense';
  if (layerName.includes('flatten')) return 'Flatten';
  if (layerName.includes('max_pooling2d')) return 'MaxPooling2D';
  return 'Unknown';
}

async function visualizeModelLayers(nextModel) {
  await nextModel.ready;
  layerNodeMap = [];

  nextModel.layers.forEach((layer, layerIndex) => {
    const config = layer.getConfig();
    const layerType = getLayerType(layer.name);

    let neuronCount = 0;
    if (layerType === 'Dense') neuronCount = config.units;
    else if (layerType === 'Conv2D') neuronCount = config.filters;
    else if (layerType === 'Flatten' || layerType === 'MaxPooling2D') {
      const outputShape = layer.outputShape;
      neuronCount = Array.isArray(outputShape) ? outputShape[outputShape.length - 1] : 0;
      neuronCount = Math.min(neuronCount, MAX_PREVIEW_NODES);
    }

    if (!neuronCount) return;
    const layerNodes = createNeuronsForLayer(layerIndex, neuronCount, layerType, layer.name);
    layerNodeMap[layerIndex] = layerNodes;
  });
}

function createNeuronsForLayer(layerIndex, neuronCount, layerType, layerName) {
  const geometry = new T.SphereGeometry(0.8, 18, 18);
  const layerNodes = [];

  for (let i = 0; i < neuronCount; i++) {
    let material = denseMat;
    if (layerType === 'Flatten') material = flattenMat;
    else if (layerType === 'Conv2D') material = conv2dMat;
    else if (layerType === 'MaxPooling2D') material = maxPoolingMat;

    const sphere = new T.Mesh(geometry, material);
    const yBase = 50 - layerIndex * 18;

    if (layerType === 'Flatten') {
      const cols = Math.ceil(Math.sqrt(neuronCount));
      const rows = Math.ceil(neuronCount / cols);
      const col = i % cols;
      const row = Math.floor(i / cols);
      const spacing = 2;
      sphere.position.x = (col - (cols - 1) / 2) * spacing;
      sphere.position.y = yBase - (row - (rows - 1) / 2) * spacing;
    } else {
      sphere.position.x = -Math.min(140, neuronCount * 2) / 2 + i * 2;
      sphere.position.y = yBase;
    }

    sphere.userData = { layerIndex, neuronIndex: i, layerType, layerName };
    nodes.add(sphere);
    scene.add(sphere);
    layerNodes.push(sphere);
  }

  return layerNodes;
}

function clearLines() {
  scene.children
    .filter(child => child.userData && child.userData.layerType === 'line')
    .forEach(line => scene.remove(line));
}

function drawDenseConnections(neuron) {
  const denseLayer = model.layers.find(l => l.name.includes('dense'));
  if (!denseLayer) return;
  const denseKernel = modelWeights.get(denseLayer.name);
  if (!denseKernel) return;

  const [inputCount, outputCount] = denseKernel.shape;
  const weights = denseKernel.weights;

  const isFlattenToDense = neuron.userData.layerType === 'Flatten';
  const isDense = neuron.userData.layerType === 'Dense';
  if (!isFlattenToDense && !isDense) return;

  if (isFlattenToDense && neuron.userData.neuronIndex >= inputCount) return;
  if (isDense && neuron.userData.neuronIndex >= outputCount) return;

  const fromIndex = neuron.userData.neuronIndex;
  const nextCount = isFlattenToDense ? outputCount : inputCount;

  for (let i = 0; i < nextCount; i++) {
    const target = scene.children.find(child => child.userData &&
      child.userData.layerType !== 'line' &&
      child.userData.layerIndex === (isDense ? neuron.userData.layerIndex - 1 : neuron.userData.layerIndex + 1) &&
      child.userData.neuronIndex === i);
    if (!target) continue;

    const weight = isFlattenToDense
      ? weights[fromIndex * outputCount + i]
      : weights[i * outputCount + fromIndex];

    const magnitude = Math.min(1, Math.abs(weight) * 4);
    const color = weight >= 0 ? new T.Color(0x2e9b37) : new T.Color(0xbe2a2a);
    const material = new T.LineBasicMaterial({ color, transparent: true, opacity: 0.2 + magnitude * 0.8 });
    const line = new T.Line(new T.BufferGeometry().setFromPoints([neuron.position, target.position]), material);
    line.userData = { layerType: 'line' };
    scene.add(line);
  }
}

function normalizeFilter(filter) {
  const flatWeights = filter.flat();
  const max = Math.max(...flatWeights);
  const min = Math.min(...flatWeights);
  if (max === min) return filter.map(row => row.map(() => 127));
  return filter.map(row => row.map(value => ((value - min) / (max - min) * 255)));
}

function displayFilter(filter, title) {
  const card = document.createElement('div');
  card.className = 'card';
  const stats = filter.flat();
  card.innerHTML = `<div><strong>${title}</strong></div><div>min=${Math.min(...stats).toFixed(1)}, max=${Math.max(...stats).toFixed(1)}</div>`;

  const canvas = document.createElement('canvas');
  const scale = 18;
  canvas.width = filter[0].length * scale;
  canvas.height = filter.length * scale;
  const ctx = canvas.getContext('2d');

  filter.forEach((row, i) => row.forEach((value, j) => {
    ctx.fillStyle = `rgb(${value},${value},${value})`;
    ctx.fillRect(j * scale, i * scale, scale, scale);
  }));

  card.appendChild(canvas);
  ui.filtersContainer.appendChild(card);
}

function drawWeights(neuron) {
  clearLines();
  ui.filtersContainer.innerHTML = '<strong>Filter Preview</strong>';
  if (neuron.userData.layerType === 'Flatten' || neuron.userData.layerType === 'Dense') {
    drawDenseConnections(neuron);
    return;
  }

  if (neuron.userData.layerType === 'Conv2D') {
    const weightsInfo = modelWeights.get(neuron.userData.layerName);
    if (!weightsInfo) return;

    const weights = weightsInfo.weights;
    const filterIndex = neuron.userData.neuronIndex;
    const outChannels = weights[0][0][0].length;
    if (filterIndex >= outChannels) return;

    const filterWeights = weights.map(row => row.map(col => col[0][filterIndex]));
    displayFilter(normalizeFilter(filterWeights), `Filter ${filterIndex + 1}`);
  }
}

function getBoundingBox(tensor2d, threshold = 0.1) {
  const values = tensor2d.dataSync();
  const h = tensor2d.shape[0];
  const w = tensor2d.shape[1];
  let minX = w;
  let maxX = -1;
  let minY = h;
  let maxY = -1;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (values[y * w + x] > threshold) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (maxX < minX || maxY < minY) return null;
  return { minX, maxX, minY, maxY };
}

function preprocessDrawingForModel(canvasEl) {
  return tf.tidy(() => {
    const base = tf.browser.fromPixels(canvasEl, 1).toFloat().div(255);
    const image2d = base.squeeze(); // [H, W]

    const box = getBoundingBox(image2d, 0.08);
    let cropped = image2d;

    if (box) {
      const width = box.maxX - box.minX + 1;
      const height = box.maxY - box.minY + 1;
      cropped = image2d.slice([box.minY, box.minX], [height, width]);
    }

    const expanded = cropped.expandDims(-1);
    const resized = tf.image.resizeBilinear(expanded, [20, 20]);
    const padded = tf.pad(resized, [[4, 4], [4, 4], [0, 0]]); // [28, 28, 1]

    return padded.expandDims(0); // [1, 28, 28, 1]
  });
}

function updateResizedPreview() {
  const preview = tf.tidy(() => {
    const t = preprocessDrawingForModel(ui.drawCanvas);
    return t.squeeze().mul(255).toInt();
  });

  if (ui.topInputPreview) {
    tf.browser.toPixels(preview, ui.topInputPreview);
  }
  tf.browser.toPixels(preview, ui.resizedCanvas).finally(() => preview.dispose());
}

function showTopPredictions(logits) {
  const probs = Array.from(logits.dataSync());
  const ranked = probs
    .map((p, i) => ({ i, p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 3)
    .map(r => `${classNames[r.i]}: ${(r.p * 100).toFixed(1)}%`);
  ui.predictionText.textContent = ranked.join(' | ');
}

function getLayerMeansForInput(inputTensor) {
  const outputs = [];
  for (const layer of model.layers) {
    outputs.push(layer.output);
  }
  const activationModel = tf.model({ inputs: model.inputs, outputs });

  const outputTensors = activationModel.predict(inputTensor);
  const tensors = Array.isArray(outputTensors) ? outputTensors : [outputTensors];

  const means = tensors.map(t => {
    const arr = t.abs().mean([0, 1, 2]).dataSync();
    t.dispose();
    return arr[0] || 0;
  });

  activationModel.dispose();
  return means;
}

function renderPredictionLayerLines(layerMeans) {
  clearLines();
  for (let i = 0; i < layerNodeMap.length - 1; i++) {
    const current = layerNodeMap[i] || [];
    const next = layerNodeMap[i + 1] || [];
    if (!current.length || !next.length) continue;

    const sampleCurrent = current.slice(0, Math.min(current.length, 16));
    const sampleNext = next.slice(0, Math.min(next.length, 16));
    const strength = Math.min(1, Math.max(layerMeans[i] || 0, layerMeans[i + 1] || 0) * 2);

    for (const fromNode of sampleCurrent) {
      for (const toNode of sampleNext) {
        const material = new T.LineBasicMaterial({
          color: new T.Color(0x3366ff),
          transparent: true,
          opacity: 0.08 + strength * 0.35
        });
        const line = new T.Line(
          new T.BufferGeometry().setFromPoints([fromNode.position, toNode.position]),
          material
        );
        line.userData = { layerType: 'line' };
        scene.add(line);
      }
    }
  }
}

function draw(ctx, x, y) {
  ctx.fillStyle = eraseMode ? 'black' : 'white';
  ctx.beginPath();
  ctx.arc(x, y, brushSize, 0, Math.PI * 2);
  ctx.fill();
}

function setupDrawingTools() {
  const ctx = ui.drawCanvas.getContext('2d');
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, ui.drawCanvas.width, ui.drawCanvas.height);

  const clearButton = document.createElement('button');
  clearButton.textContent = 'Clear (C)';
  clearButton.onclick = () => {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, ui.drawCanvas.width, ui.drawCanvas.height);
    ui.predictionText.textContent = 'No prediction yet.';
    clearLines();
    updateResizedPreview();
  };

  const inspectButton = document.createElement('button');
  inspectButton.textContent = 'Inspect mode: OFF';
  inspectButton.onclick = () => {
    inspectMode = !inspectMode;
    inspectButton.textContent = `Inspect mode: ${inspectMode ? 'ON' : 'OFF'}`;
  };

  const predictButton = document.createElement('button');
  predictButton.textContent = 'Predict (P)';
  predictButton.onclick = () => {
    const input = preprocessDrawingForModel(ui.drawCanvas);
    const prediction = model.predict(input);
    const logits = prediction.squeeze();

    showTopPredictions(logits);
    const layerMeans = getLayerMeansForInput(input);
    renderPredictionLayerLines(layerMeans);

    logits.dispose();
    input.dispose();
    prediction.dispose();
    updateResizedPreview();
  };

  const resizeButton = document.createElement('button');
  resizeButton.textContent = 'Update 28x28 Preview';
  resizeButton.onclick = updateResizedPreview;

  const resetCameraButton = document.createElement('button');
  resetCameraButton.textContent = 'Reset Camera';
  resetCameraButton.onclick = () => {
    camera.position.set(0, 0, 100);
    controls.target.set(0, 0, 0);
    controls.update();
  };

  const eraseButton = document.createElement('button');
  eraseButton.textContent = 'Mode: Draw';
  eraseButton.onclick = () => {
    eraseMode = !eraseMode;
    eraseButton.textContent = `Mode: ${eraseMode ? 'Erase' : 'Draw'}`;
  };

  const downloadButton = document.createElement('button');
  downloadButton.textContent = 'Download Model';
  downloadButton.onclick = async () => { await model.save('downloads://my-model'); };

  const brush = document.createElement('input');
  brush.type = 'range';
  brush.min = '4';
  brush.max = '24';
  brush.value = String(brushSize);
  brush.oninput = () => { brushSize = Number(brush.value); };

  ui.buttonRow.append(inspectButton, resetCameraButton, downloadButton);
  ui.drawTools.append(clearButton, eraseButton, predictButton, resizeButton, brush);

  const pointerToDraw = (event) => {
    if (!drawing) return;
    const rect = ui.drawCanvas.getBoundingClientRect();
    draw(ctx, event.clientX - rect.left, event.clientY - rect.top);
    updateResizedPreview();
  };

  ui.drawCanvas.addEventListener('pointerdown', () => { drawing = true; inspectMode = false; });
  ui.drawCanvas.addEventListener('pointerup', () => { drawing = false; });
  ui.drawCanvas.addEventListener('pointerleave', () => { drawing = false; });
  ui.drawCanvas.addEventListener('pointermove', pointerToDraw);

  window.addEventListener('keydown', e => {
    if (e.key.toLowerCase() === 'c') clearButton.click();
    if (e.key.toLowerCase() === 'p') predictButton.click();
  });

  updateResizedPreview();
}

function setupThree() {
  scene = new T.Scene();
  camera = new T.PerspectiveCamera(60, 1, 0.1, 2000);
  camera.position.set(0, 0, 100);
  scene.add(camera);
  nodes = new T.Group();

  renderer = new T.WebGLRenderer({ antialias: true });
  ui.sceneCanvasWrap.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const resize = () => {
    const w = ui.sceneCanvasWrap.clientWidth;
    const h = ui.sceneCanvasWrap.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  resize();
  window.addEventListener('resize', resize);

  renderer.domElement.addEventListener('mousemove', event => {
    if (!inspectMode) return;
    const bounds = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
    mouse.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children);
    if (intersects.length > 0) {
      const neuron = intersects[0].object;
      if (neuron.userData && neuron.userData.layerType !== 'line') drawWeights(neuron);
    }
  });
}

async function doPrediction(nextModel, data, testDataSize = 500) {
  const testData = data.nextTestBatch(testDataSize);
  const [preds, labels] = tf.tidy(() => {
    const testxs = testData.xs.reshape([testDataSize, 28, 28, 1]);
    const labels = testData.labels.argMax(-1);
    const preds = nextModel.predict(testxs).argMax(-1);
    return [preds, labels];
  });
  testData.xs.dispose();
  testData.labels.dispose();
  return [preds, labels];
}

async function showAccuracy(nextModel, data) {
  const [preds, labels] = await doPrediction(nextModel, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  tfvis.show.perClassAccuracy({ name: 'Accuracy', tab: 'Evaluation' }, classAccuracy, classNames);
  preds.dispose();
  labels.dispose();
}

async function showConfusion(nextModel, data) {
  const [preds, labels] = await doPrediction(nextModel, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  tfvis.render.confusionMatrix({ name: 'Confusion Matrix', tab: 'Evaluation' }, { values: confusionMatrix, tickLabels: classNames });
  preds.dispose();
  labels.dispose();
}

async function showExamples(data) {
  const surface = tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data' });
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => examples.xs.slice([i, 0], [1, examples.xs.shape[1]]).reshape([28, 28, 1]));
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style.margin = '4px';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);
    imageTensor.dispose();
  }
  examples.xs.dispose();
  examples.labels.dispose();
}

async function run() {
  setupThree();
  setupDrawingTools();

  setStatus('Loading data...');
  setBusyState(true);
  const data = new MnistData();
  await data.load();
  await showExamples(data);

  try {
    setStatus('Loading pre-trained model...');
    model = await tf.loadLayersModel('./my-model.json');
    updateModelStatus('Loaded: my-model.json');
  } catch (error) {
    console.error(error);
    updateModelStatus('Pre-trained model unavailable. Training fallback model...');
    model = getModel();
    await train(model, data);
    updateModelStatus('Fallback model trained in-browser.');
  }

  tfvis.show.modelSummary({ name: 'Model Architecture' }, model);
  modelWeights = await extractWeights(model);
  await visualizeModelLayers(model);

  setStatus('Running evaluation...');
  await showAccuracy(model, data);
  await showConfusion(model, data);

  if (DEBUG_MEMORY) {
    console.log('tf.memory()', tf.memory());
  }

  setBusyState(false);
  setStatus('Ready. Draw a digit, then click Predict to render output and layer links.');
  animate();
}

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

(function normalizationSelfCheck() {
  const single = normalizeFilter([[2, 2], [2, 2]]);
  if (single.flat().some(Number.isNaN)) {
    console.error('normalizeFilter produced NaN values on zero-variance input.');
  }
})();
