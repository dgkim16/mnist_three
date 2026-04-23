import * as T from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {MnistData} from './data.js';

const classNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

// ---- 3D voxel layout constants ----
//
// Every feature map is rendered as a 3-axis voxel block:
//   Y axis = row (H)      — vertical in world space
//   Z axis = col (W)      — depth in world space
//   X axis = layer origin + channel offset  — channels are stacked slabs
//
// Because row, col, and channel each get their own world axis, rotating the
// view always reveals a new spatial relationship. This is the property the
// plane-per-channel version lacked.
const SPATIAL_SPACING = 0.065;
const CHANNEL_SPACING = 0.22;
const CUBE_SIZE = 0.055;

// Dim slate-blue for resting voxels — bright enough to stand out against the
// very dark card background (0x0a0c10) after the ambient/diffuse chain (Three
// renders in linear space by default), but dim enough that activated voxels
// in VOXEL_ACCENT still pop.
const VOXEL_BG = new T.Color(0.38, 0.42, 0.5);
const VOXEL_ACCENT = new T.Color(1.0, 0.54, 0.24);
const KERNEL_POS = new T.Color(1.0, 0.54, 0.24);
const KERNEL_NEG = new T.Color(0.28, 0.56, 1.0);

// Metadata for every layer whose activations we visualize. Populated at load;
// each entry owns one canvas per channel (for the 2D grid) and one shared
// InstancedMesh voxel block (for the 3D view).
//
// Shape: [{name, displayName, H, W, C, x3d,
//          sourceFn(pre, acts)→tensor,
//          channels: [{canvas}],
//          instancedMesh: T.InstancedMesh}]
const layerViews = [];

let model;
let activationModel;
let data = null;
let devPanelReady = false;

// Three.js state
let renderer3d, scene3d, camera3d, controls3d;
let raycaster3d, mouse3d;
let hoverLabelEl;
let voxelMeshes = [];
let meshToView = new Map();
let receptiveFieldGroup = null;
let kernelInspectorGroup = null;
let lastHover = null;
let convKernels = null; // {conv1: [5][5][1][8], conv2: [5][5][8][16]}

document.addEventListener('DOMContentLoaded', run);

async function run() {
    model = await tf.loadLayersModel('./my-model.json');

    // Multi-output model that returns every layer's activation in one pass.
    activationModel = tf.model({
        inputs: model.inputs,
        outputs: model.layers.map(l => l.output),
    });

    tfvis.visor().close();

    extractConvKernels();
    buildLayerViews();
    buildActivationPanel();
    build3DScene();
    await renderConv1Filters();
    setupUI();

    document.getElementById('loading').style.display = 'none';
    document.getElementById('app').style.display = 'grid';

    // Size the 3D canvas once the layout has been resolved.
    resize3D();
}

// Pull the two conv layers' kernels into JS arrays once, for the hover
// kernel-inspector to read per-filter slices without hitting WebGL each frame.
function extractConvKernels() {
    const conv1W = model.layers[0].getWeights()[0]; // [5, 5, 1, 8]
    const conv2W = model.layers[2].getWeights()[0]; // [5, 5, 8, 16]
    convKernels = {
        conv1: conv1W.arraySync(),
        conv2: conv2W.arraySync(),
    };
}

// Build per-layer metadata with persistent per-channel canvases used by the
// 2D activation grid. The 3D view uses an InstancedMesh built separately in
// build3DScene().
function buildLayerViews() {
    // activationModel output order matches model.layers:
    //   [0] conv1, [1] pool1, [2] conv2, [3] flatten, [4] dense
    const descriptors = [
        {name: 'input',    displayName: 'input (pre-processed)',    H: 28, W: 28, C: 1,
            x3d: -3.6, sourceFn: (pre, _) => pre},
        {name: 'conv1',    displayName: model.layers[0].name,       H: 24, W: 24, C: 8,
            x3d: -1.4, sourceFn: (_, a) => a[0]},
        {name: 'pool1',    displayName: model.layers[1].name,       H: 12, W: 12, C: 8,
            x3d:  0.8, sourceFn: (_, a) => a[1]},
        {name: 'conv2',    displayName: model.layers[2].name,       H:  8, W:  8, C: 16,
            x3d:  3.6, sourceFn: (_, a) => a[2]},
    ];
    for (const d of descriptors) {
        const channels = [];
        for (let c = 0; c < d.C; c++) {
            const canvas = document.createElement('canvas');
            canvas.width = d.W;
            canvas.height = d.H;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, d.W, d.H);
            channels.push({canvas});
        }
        layerViews.push({...d, channels, instancedMesh: null});
    }
}

function buildActivationPanel() {
    const container = document.getElementById('activations');
    container.innerHTML = '';

    for (const view of layerViews) {
        const wrap = document.createElement('div');
        wrap.className = 'layer';

        const head = document.createElement('div');
        head.className = 'layer-head';
        const name = document.createElement('div');
        name.className = 'name';
        name.textContent = view.displayName;
        const shape = document.createElement('div');
        shape.className = 'shape';
        shape.textContent = `${view.H} × ${view.W} × ${view.C}`;
        head.appendChild(name);
        head.appendChild(shape);
        wrap.appendChild(head);

        const grid = document.createElement('div');
        grid.className = 'channels';
        for (const ch of view.channels) {
            grid.appendChild(ch.canvas);
        }
        wrap.appendChild(grid);
        container.appendChild(wrap);
    }

    // Static note for flatten / dense — the softmax bar chart already covers
    // their semantic output; this just records the tensor shapes.
    for (let i = 3; i < model.layers.length; i++) {
        const layer = model.layers[i];
        const units = layer.outputShape[layer.outputShape.length - 1];
        const wrap = document.createElement('div');
        wrap.className = 'layer';
        const head = document.createElement('div');
        head.className = 'layer-head';
        const n = document.createElement('div');
        n.className = 'name';
        n.textContent = layer.name;
        const s = document.createElement('div');
        s.className = 'shape';
        s.textContent = `${units} values`;
        head.appendChild(n);
        head.appendChild(s);
        wrap.appendChild(head);
        const note = document.createElement('div');
        note.className = 'layer-note';
        note.textContent = i === model.layers.length - 1
            ? '(see Prediction panel)'
            : '';
        if (note.textContent) wrap.appendChild(note);
        container.appendChild(wrap);
    }
}

function build3DScene() {
    const container = document.getElementById('network3d');
    hoverLabelEl = document.getElementById('hoverLabel3d');

    renderer3d = new T.WebGLRenderer({antialias: true, alpha: false});
    renderer3d.setPixelRatio(window.devicePixelRatio);
    renderer3d.setClearColor(0x0a0c10, 1);
    container.appendChild(renderer3d.domElement);

    scene3d = new T.Scene();

    camera3d = new T.PerspectiveCamera(40, 1, 0.1, 100);
    camera3d.position.set(1.5, 2.2, 9.0);

    controls3d = new OrbitControls(camera3d, renderer3d.domElement);
    controls3d.enableDamping = true;
    controls3d.dampingFactor = 0.08;
    controls3d.target.set(0.6, 0, 0);
    controls3d.minDistance = 3;
    controls3d.maxDistance = 22;

    // Lights so voxel box faces shade differently; unlit voxels look 2D.
    scene3d.add(new T.AmbientLight(0xffffff, 0.55));
    const dirA = new T.DirectionalLight(0xffffff, 0.8);
    dirA.position.set(3, 5, 6);
    scene3d.add(dirA);
    const dirB = new T.DirectionalLight(0xa0b4d0, 0.28);
    dirB.position.set(-4, -2, -3);
    scene3d.add(dirB);

    raycaster3d = new T.Raycaster();
    mouse3d = new T.Vector2(-10, -10);

    voxelMeshes = [];
    meshToView = new Map();

    const dummy = new T.Object3D();
    const initColor = new T.Color().copy(VOXEL_BG);

    for (const view of layerViews) {
        const count = view.H * view.W * view.C;
        const geom = new T.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
        const mat = new T.MeshLambertMaterial({color: 0xffffff});
        const mesh = new T.InstancedMesh(geom, mat, count);

        for (let c = 0; c < view.C; c++) {
            for (let y = 0; y < view.H; y++) {
                for (let x = 0; x < view.W; x++) {
                    const i = instanceIdOf(view, c, y, x);
                    const p = voxelWorldPos(view, c, y, x);
                    dummy.position.set(p.x, p.y, p.z);
                    dummy.rotation.set(0, 0, 0);
                    dummy.updateMatrix();
                    mesh.setMatrixAt(i, dummy.matrix);
                    mesh.setColorAt(i, initColor);
                }
            }
        }
        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        mesh.userData = {viewName: view.name};

        scene3d.add(mesh);
        view.instancedMesh = mesh;
        voxelMeshes.push(mesh);
        meshToView.set(mesh, view);
    }

    // Groups for hover-driven overlays (receptive field + kernel inspector).
    // Rebuilt on every hover change.
    receptiveFieldGroup = new T.Group();
    kernelInspectorGroup = new T.Group();
    scene3d.add(receptiveFieldGroup);
    scene3d.add(kernelInspectorGroup);

    renderer3d.domElement.addEventListener('pointermove', event => {
        const rect = renderer3d.domElement.getBoundingClientRect();
        mouse3d.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse3d.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    });
    renderer3d.domElement.addEventListener('pointerleave', () => {
        mouse3d.set(-10, -10);
        hoverLabelEl.style.display = 'none';
        clearHoverVisuals();
        lastHover = null;
    });

    const ro = new ResizeObserver(() => resize3D());
    ro.observe(container);

    animate3D();
}

function resize3D() {
    if (!renderer3d) return;
    const container = document.getElementById('network3d');
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (width === 0 || height === 0) return;
    renderer3d.setSize(width, height);
    camera3d.aspect = width / height;
    camera3d.updateProjectionMatrix();
}

// World position of voxel (c, y, x) in a given layer. Y flipped so row 0
// sits at the top of the visual frame.
function voxelWorldPos(view, c, y, x) {
    const centerC = (view.C - 1) / 2;
    const centerH = (view.H - 1) / 2;
    const centerW = (view.W - 1) / 2;
    return new T.Vector3(
        view.x3d + (c - centerC) * CHANNEL_SPACING,
        -(y - centerH) * SPATIAL_SPACING,
        (x - centerW) * SPATIAL_SPACING,
    );
}

function instanceIdOf(view, c, y, x) {
    return c * view.H * view.W + y * view.W + x;
}

function animate3D() {
    requestAnimationFrame(animate3D);
    controls3d.update();

    raycaster3d.setFromCamera(mouse3d, camera3d);
    const hits = raycaster3d.intersectObjects(voxelMeshes, false);
    if (hits.length > 0) {
        const mesh = hits[0].object;
        const view = meshToView.get(mesh);
        const id = hits[0].instanceId;
        const c = Math.floor(id / (view.H * view.W));
        const rest = id % (view.H * view.W);
        const y = Math.floor(rest / view.W);
        const x = rest % view.W;

        hoverLabelEl.textContent = `${view.displayName}  ·  ch ${c}  ·  (y=${y}, x=${x})`;
        hoverLabelEl.style.display = 'block';

        if (!lastHover
            || lastHover.viewName !== view.name
            || lastHover.c !== c
            || lastHover.y !== y
            || lastHover.x !== x) {
            clearHoverVisuals();
            drawReceptiveField(view, c, y, x);
            drawKernelInspector(view, c, y, x);
            lastHover = {viewName: view.name, c, y, x};
        }
    } else {
        hoverLabelEl.style.display = 'none';
        if (lastHover) {
            clearHoverVisuals();
            lastHover = null;
        }
    }

    renderer3d.render(scene3d, camera3d);
}

// For the hovered voxel, draw line segments back to the voxels in the previous
// layer that actually feed into it — i.e. a rendering of the real receptive
// field, as a function of layer geometry.
function drawReceptiveField(view, c, y, x) {
    const target = voxelWorldPos(view, c, y, x);
    const segs = [];
    const lineMat = new T.LineBasicMaterial({
        color: 0xff8a3d,
        transparent: true,
        opacity: 0.45,
    });

    if (view.name === 'conv1') {
        // 5×5 receptive field from input (single channel).
        const input = layerViews[0];
        for (let dy = 0; dy < 5; dy++) {
            for (let dx = 0; dx < 5; dx++) {
                const src = voxelWorldPos(input, 0, y + dy, x + dx);
                segs.push(target.x, target.y, target.z, src.x, src.y, src.z);
            }
        }
    } else if (view.name === 'pool1') {
        // 2×2 from conv1 (same channel — max pool doesn't mix channels).
        const conv1 = layerViews[1];
        for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
                const src = voxelWorldPos(conv1, c, 2 * y + dy, 2 * x + dx);
                segs.push(target.x, target.y, target.z, src.x, src.y, src.z);
            }
        }
    } else if (view.name === 'conv2') {
        // 5×5×8 receptive field from pool1 = 200 lines. Draw a wireframe
        // hull over the 5×5 spatial patch across all 8 pool1 channels, plus
        // one line from the hovered voxel to the hull centre.
        const pool1 = layerViews[2];
        const a = voxelWorldPos(pool1, 0, y, x);
        const b = voxelWorldPos(pool1, pool1.C - 1, y + 4, x + 4);
        const centre = new T.Vector3(
            (a.x + b.x) / 2,
            (a.y + b.y) / 2,
            (a.z + b.z) / 2,
        );
        const sx = Math.abs(b.x - a.x) + CUBE_SIZE * 2;
        const sy = Math.abs(a.y - b.y) + CUBE_SIZE * 2;
        const sz = Math.abs(b.z - a.z) + CUBE_SIZE * 2;
        const boxGeom = new T.BoxGeometry(sx, sy, sz);
        const edges = new T.EdgesGeometry(boxGeom);
        const hull = new T.LineSegments(edges, new T.LineBasicMaterial({
            color: 0xff8a3d,
            transparent: true,
            opacity: 0.55,
        }));
        hull.position.copy(centre);
        receptiveFieldGroup.add(hull);
        boxGeom.dispose();

        segs.push(target.x, target.y, target.z, centre.x, centre.y, centre.z);
    }

    if (segs.length > 0) {
        const geom = new T.BufferGeometry();
        geom.setAttribute('position', new T.Float32BufferAttribute(segs, 3));
        const lines = new T.LineSegments(geom, lineMat);
        receptiveFieldGroup.add(lines);
    }
}

// Materialise the learned convolution kernel for the hovered filter as a
// floating 3D voxel volume. Orange = positive weight, blue = negative,
// opacity = |weight| / maxMagnitude. For Conv2 this is a real 5×5×8 volume —
// one slice per input channel — which is itself an inherently-3D object.
function drawKernelInspector(view, c, y, x) {
    if (!convKernels) return;

    let kernel;
    let inC;
    if (view.name === 'conv1') {
        // conv1 kernel shape [5][5][1][8] — filter c, single input channel.
        const k = convKernels.conv1;
        inC = 1;
        kernel = new Array(5);
        for (let ky = 0; ky < 5; ky++) {
            kernel[ky] = new Array(5);
            for (let kx = 0; kx < 5; kx++) {
                kernel[ky][kx] = [k[ky][kx][0][c]];
            }
        }
    } else if (view.name === 'conv2') {
        // conv2 kernel shape [5][5][8][16] — filter c, all 8 input channels.
        const k = convKernels.conv2;
        inC = 8;
        kernel = new Array(5);
        for (let ky = 0; ky < 5; ky++) {
            kernel[ky] = new Array(5);
            for (let kx = 0; kx < 5; kx++) {
                kernel[ky][kx] = new Array(inC);
                for (let ci = 0; ci < inC; ci++) {
                    kernel[ky][kx][ci] = k[ky][kx][ci][c];
                }
            }
        }
    } else {
        return; // no kernel for input / pool
    }

    let maxMag = 1e-9;
    for (let ky = 0; ky < 5; ky++) {
        for (let kx = 0; kx < 5; kx++) {
            for (let ci = 0; ci < inC; ci++) {
                const m = Math.abs(kernel[ky][kx][ci]);
                if (m > maxMag) maxMag = m;
            }
        }
    }

    // Inspector floats above the hovered layer, centred on its x3d.
    const inspX = view.x3d;
    const inspY = 1.55;
    const inspZ = 0;
    const inspSpacing = 0.11;
    const inspChSpacing = inC === 1 ? 0 : 0.5;
    const cubeSize = 0.09;

    for (let ky = 0; ky < 5; ky++) {
        for (let kx = 0; kx < 5; kx++) {
            for (let ci = 0; ci < inC; ci++) {
                const w = kernel[ky][kx][ci];
                const mag = Math.min(1, Math.abs(w) / maxMag);
                const colour = w >= 0 ? KERNEL_POS : KERNEL_NEG;
                const mat = new T.MeshLambertMaterial({
                    color: colour,
                    transparent: true,
                    opacity: 0.14 + 0.82 * mag,
                });
                const geom = new T.BoxGeometry(cubeSize, cubeSize, cubeSize);
                const box = new T.Mesh(geom, mat);
                box.position.set(
                    inspX + (ci - (inC - 1) / 2) * inspChSpacing,
                    inspY - (ky - 2) * inspSpacing,
                    inspZ + (kx - 2) * inspSpacing,
                );
                kernelInspectorGroup.add(box);
            }
        }
    }

    // Faint connector from the hovered voxel up to the inspector.
    const anchor = voxelWorldPos(view, c, y, x);
    const linkGeom = new T.BufferGeometry().setFromPoints([
        anchor,
        new T.Vector3(inspX, inspY - 0.35, inspZ),
    ]);
    const link = new T.Line(linkGeom, new T.LineBasicMaterial({
        color: 0xff8a3d,
        transparent: true,
        opacity: 0.35,
    }));
    kernelInspectorGroup.add(link);
}

function clearHoverVisuals() {
    disposeGroupChildren(receptiveFieldGroup);
    disposeGroupChildren(kernelInspectorGroup);
}

function disposeGroupChildren(group) {
    if (!group) return;
    while (group.children.length) {
        const child = group.children[0];
        group.remove(child);
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
            if (Array.isArray(child.material)) {
                child.material.forEach(m => m.dispose());
            } else {
                child.material.dispose();
            }
        }
    }
}

function setupUI() {
    const drawCanvas = document.getElementById('drawCanvas');
    const ctx = drawCanvas.getContext('2d');
    resetCanvas(ctx);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    let drawing = false;
    let drawRect = null;
    let lastX = 0, lastY = 0;

    drawCanvas.addEventListener('pointerdown', event => {
        drawing = true;
        drawCanvas.setPointerCapture(event.pointerId);
        drawRect = drawCanvas.getBoundingClientRect();
        lastX = event.clientX - drawRect.left;
        lastY = event.clientY - drawRect.top;
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(lastX, lastY, 10, 0, Math.PI * 2);
        ctx.fill();
    });
    drawCanvas.addEventListener('pointermove', event => {
        if (!drawing) return;
        const x = event.clientX - drawRect.left;
        const y = event.clientY - drawRect.top;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        lastX = x;
        lastY = y;
    });
    const endStroke = async event => {
        if (!drawing) return;
        drawing = false;
        if (drawCanvas.hasPointerCapture(event.pointerId)) {
            drawCanvas.releasePointerCapture(event.pointerId);
        }
        await runInference(ctx);
    };
    drawCanvas.addEventListener('pointerup', endStroke);
    drawCanvas.addEventListener('pointercancel', endStroke);

    document.getElementById('predictBtn').addEventListener('click', () => runInference(ctx));
    document.getElementById('clearBtn').addEventListener('click', () => {
        resetCanvas(ctx);
        resetUI();
    });
    document.getElementById('downloadBtn').addEventListener('click', async () => {
        await model.save('downloads://my-model');
    });
    document.getElementById('devBtn').addEventListener('click', openDevPanel);
}

function resetCanvas(ctx) {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 280, 280);
    ctx.fillStyle = 'white';
}

function resetUI() {
    const big = document.getElementById('prediction-big');
    big.textContent = 'Draw a digit';
    big.classList.add('empty');
    document.getElementById('bars').innerHTML = '';

    const preview = document.getElementById('previewCanvas');
    const pctx = preview.getContext('2d');
    pctx.fillStyle = 'black';
    pctx.fillRect(0, 0, 28, 28);

    const zero = new T.Color().copy(VOXEL_BG);
    for (const view of layerViews) {
        for (const ch of view.channels) {
            const cctx = ch.canvas.getContext('2d');
            cctx.fillStyle = 'black';
            cctx.fillRect(0, 0, view.W, view.H);
        }
        if (view.instancedMesh) {
            const count = view.H * view.W * view.C;
            for (let i = 0; i < count; i++) {
                view.instancedMesh.setColorAt(i, zero);
            }
            if (view.instancedMesh.instanceColor) {
                view.instancedMesh.instanceColor.needsUpdate = true;
            }
        }
    }

    clearHoverVisuals();
    lastHover = null;
}

async function runInference(ctx) {
    const pre = preprocessForMnist(ctx);

    const previewTensor = pre.squeeze([0]);
    await tf.browser.toPixels(previewTensor, document.getElementById('previewCanvas'));
    previewTensor.dispose();

    const activations = activationModel.predict(pre);
    const softmax = activations[activations.length - 1];
    const probs = await softmax.data();

    renderPrediction(probs);
    await updateFeatureMaps(pre, activations);

    for (const t of activations) t.dispose();
    pre.dispose();
}

function renderPrediction(probs) {
    const big = document.getElementById('prediction-big');
    const bars = document.getElementById('bars');

    let topIdx = 0;
    for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[topIdx]) topIdx = i;
    }
    big.textContent = classNames[topIdx];
    big.classList.remove('empty');

    bars.innerHTML = '';
    for (let i = 0; i < probs.length; i++) {
        const pct = (probs[i] * 100).toFixed(1);
        const row = document.createElement('div');
        row.className = 'bar-row' + (i === topIdx ? ' top' : '');

        const label = document.createElement('div');
        label.textContent = classNames[i];

        const track = document.createElement('div');
        track.className = 'bar-track';
        const fill = document.createElement('div');
        fill.className = 'bar-fill';
        fill.style.width = pct + '%';
        track.appendChild(fill);

        const value = document.createElement('div');
        value.className = 'bar-value';
        value.textContent = pct + '%';

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(value);
        bars.appendChild(row);
    }
}

// Write per-channel activations to the persistent canvases (2D panel) and to
// the InstancedMesh per-voxel colours (3D view). Channels are normalized
// independently (per-channel min/max) so low-magnitude feature maps still
// show internal structure.
//
// Perf: one GPU->CPU read per layer (not per channel) — the normalized
// [H, W, C] tensor is pulled to JS once, then used to write the 2D canvases
// (via putImageData, no second GPU read) and the 3D voxel colours.
async function updateFeatureMaps(pre, activations) {
    const tmpColor = new T.Color();
    for (const view of layerViews) {
        const tensor = view.sourceFn(pre, activations); // [1, H, W, C]

        // Per-channel normalization; keep channel axis for broadcasting.
        const normWHC = tf.tidy(() => {
            const sq = tensor.squeeze([0]);                  // [H, W, C]
            const mins = sq.min([0, 1], true);               // [1, 1, C]
            const maxs = sq.max([0, 1], true);
            const ranges = maxs.sub(mins).maximum(tf.scalar(1e-9));
            return sq.sub(mins).div(ranges);                 // [H, W, C] in [0, 1]
        });
        const flat = await normWHC.data(); // length H*W*C, layout (y, x, c)
        normWHC.dispose();

        const H = view.H, W = view.W, C = view.C;
        for (let c = 0; c < C; c++) {
            // 2D panel: write pixel values directly to the channel canvas.
            const ctx2 = view.channels[c].canvas.getContext('2d');
            const imgData = ctx2.createImageData(W, H);
            for (let y = 0; y < H; y++) {
                for (let xx = 0; xx < W; xx++) {
                    const v = flat[(y * W + xx) * C + c];
                    const px = Math.max(0, Math.min(255, Math.round(v * 255)));
                    const i4 = (y * W + xx) * 4;
                    imgData.data[i4] = px;
                    imgData.data[i4 + 1] = px;
                    imgData.data[i4 + 2] = px;
                    imgData.data[i4 + 3] = 255;
                }
            }
            ctx2.putImageData(imgData, 0, 0);

            // 3D voxels: lerp bg→accent by normalized activation.
            if (view.instancedMesh) {
                const base = c * H * W;
                for (let y = 0; y < H; y++) {
                    for (let xx = 0; xx < W; xx++) {
                        const v = flat[(y * W + xx) * C + c];
                        tmpColor.copy(VOXEL_BG).lerp(VOXEL_ACCENT, Math.pow(v, 0.7));
                        view.instancedMesh.setColorAt(base + y * W + xx, tmpColor);
                    }
                }
            }
        }
        if (view.instancedMesh && view.instancedMesh.instanceColor) {
            view.instancedMesh.instanceColor.needsUpdate = true;
        }
    }
}

async function renderConv1Filters() {
    const strip = document.getElementById('filterStrip');
    strip.innerHTML = '';
    const conv1 = model.layers[0];
    const weights = conv1.getWeights();
    if (!weights || weights.length === 0) return;

    const kernel = weights[0]; // [5, 5, 1, filters]
    const [kH, kW, , filters] = kernel.shape;
    const normalized = tf.tidy(() => {
        const list = [];
        for (let f = 0; f < filters; f++) {
            const slice = kernel.slice([0, 0, 0, f], [kH, kW, 1, 1]).squeeze([3]);
            const min = slice.min();
            const range = slice.max().sub(min).maximum(tf.scalar(1e-9));
            list.push(slice.sub(min).div(range));
        }
        return list;
    });
    for (const filt of normalized) {
        const canvas = document.createElement('canvas');
        canvas.width = kW;
        canvas.height = kH;
        await tf.browser.toPixels(filt, canvas);
        strip.appendChild(canvas);
        filt.dispose();
    }
}

async function openDevPanel() {
    const visor = tfvis.visor();
    visor.open();
    if (devPanelReady) return;
    devPanelReady = true;
    tfvis.show.modelSummary({name: 'Model Architecture'}, model);
    data = new MnistData();
    await data.load();
    await showExamples(data);
    await showAccuracy(model, data);
    await showConfusion(model, data);
}

// MNIST digits sit inside a ~20x20 region centred in a 28x28 frame with ~4px of
// black padding. A raw 280x280 user drawing fills the frame, which is heavily
// out-of-distribution for the trained CNN. This function crops the ink bbox,
// scales its longest side to 20, pastes it centred into 28x28, and normalizes
// pixels to [0, 1] — yielding a [1, 28, 28, 1] float tensor ready for predict.
function preprocessForMnist(sourceCtx) {
    const W = 280, H = 280;
    const imageData = sourceCtx.getImageData(0, 0, W, H);
    const pixels = imageData.data;

    let minX = W, minY = H, maxX = -1, maxY = -1;
    const threshold = 10;
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            if (pixels[(y * W + x) * 4] > threshold) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    if (maxX < 0) {
        return tf.zeros([1, 28, 28, 1]);
    }

    const bboxW = maxX - minX + 1;
    const bboxH = maxY - minY + 1;
    const scale = 20 / Math.max(bboxW, bboxH);
    const targetW = Math.max(1, Math.round(bboxW * scale));
    const targetH = Math.max(1, Math.round(bboxH * scale));
    const offsetX = Math.floor((28 - targetW) / 2);
    const offsetY = Math.floor((28 - targetH) / 2);

    const tmp = document.createElement('canvas');
    tmp.width = 28;
    tmp.height = 28;
    const tmpCtx = tmp.getContext('2d');
    tmpCtx.fillStyle = 'black';
    tmpCtx.fillRect(0, 0, 28, 28);
    tmpCtx.imageSmoothingEnabled = true;
    tmpCtx.drawImage(
        sourceCtx.canvas,
        minX, minY, bboxW, bboxH,
        offsetX, offsetY, targetW, targetH
    );

    return tf.tidy(() => tf.browser.fromPixels(tmp, 1)
        .toFloat()
        .div(255.0)
        .expandDims(0));
}

// ---- Dev panel (tfjs-vis surfaces, opt-in) ----

async function showExamples(data) {
    const surface = tfvis.visor().surface({name: 'Input Data Examples', tab: 'Input Data'});
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => examples.xs
            .slice([i, 0], [1, examples.xs.shape[1]])
            .reshape([28, 28, 1]));
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);
        imageTensor.dispose();
    }
}

function doPrediction(model, data, testDataSize = 500) {
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, 28, 28, 1]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);
    testxs.dispose();
    return [preds, labels];
}

async function showAccuracy(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    tfvis.show.perClassAccuracy({name: 'Accuracy', tab: 'Evaluation'}, classAccuracy, classNames);
    labels.dispose();
    preds.dispose();
}

async function showConfusion(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    tfvis.render.confusionMatrix(
        {name: 'Confusion Matrix', tab: 'Evaluation'},
        {values: confusionMatrix, tickLabels: classNames}
    );
    labels.dispose();
    preds.dispose();
}

// ---- Training path (unused but preserved for retraining) ----

function getModel() {
    const m = tf.sequential();
    m.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
    }));
    m.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    m.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
    }));
    m.add(tf.layers.flatten());
    m.add(tf.layers.dense({
        units: 10,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax',
    }));
    m.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    return m;
}

async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {name: 'Model Training', tab: 'Model', styles: {height: '1000px'}};
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

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

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 15,
        shuffle: true,
        callbacks: fitCallbacks,
    });
}
