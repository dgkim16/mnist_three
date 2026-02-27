console.log('Hello TensorFlow!');

import * as T from "https://unpkg.com/three@v0.149.0/build/three.module.js";
import {OrbitControls} from "https://unpkg.com/three@v0.149.0/examples/jsm/controls/OrbitControls.js";
import {MnistData} from './data.js';


async function showExamples(data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

function getModel() {
    const model = tf.sequential();
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;


    // first layer. specify input shape & convolution operations
    // purpose of filters is to extract features from the input image
    // filters are initially unknown, but are learned during training
    // variance Scaling is a common weight initialization technique
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    // Max pooling layer, downsamples using max values in a region instead of averaging
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }))

    // Repeat another conv2d + maxPooling stack
    // more filters!
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    // flatten the output from 2D filters into 1D vector to prepare for input to last layer
    // common practice when feeding high dimension data to final classification output layer
    model.add(tf.layers.flatten());

    // last layer (output layer) uses softmax to generate data between 0 and 1 (propability, sum of all classes = 1)
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }))

    // the last two layers get merged into a single layer.

    // optimizer. adam is a popular optimizer
    // loss function of categoricalCrossentropy: (minimize the loss function to improve the model)
        // Sum ( (1-y)log(1-y_hat) + ylog(y_hat) ). 
        // y is the true label, y_hat is the predicted label
    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })
    
    return model;
}

async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training', tab: 'Model', styles: {height: '1000px'}
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return[
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return[
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 15,
        shuffle: true,
        callbacks: fitCallbacks
    });
}

// APPROACH IS NOT CORRECT.
// Conv2d layer is not fully connected.
// Directly extract the weights from each conv2d layer.

// chatgpt4 generated code
// dataSync() is done, so the weight values can be accessed via .weights
async function extractWeights(model) {
    const modelWeights = [];
    model.layers.forEach(layer => {
    if (layer.getWeights().length > 0) {
        const weightsTensor = layer.getWeights()[0]; // Get the first tensor of weights, typically the kernel
        const name = layer.name;
        var weights;
        if(layer.name.includes('conv2d'))
            weights = weightsTensor.arraySync();
        else
            weights = weightsTensor.dataSync(); // Synchronously download the tensor's values
        modelWeights.push({
            weights: weights,
            shape: weightsTensor.shape, // Use the tensor's shape property directly
            name: name
        });
    }
    });
    return modelWeights;
}


// chatgpt4 used
async function visualizeModelLayers(model) {
    // Wait for the model to be ready if it's not already
    await model.ready;
    // Iterate through each layer of the model
    model.layers.forEach((layer, layerIndex) => {
        console.log(layerIndex);
        let neuronCount;

        // Attempt to get layer configuration which usually contains type information
        const config = layer.getConfig();
        
        // Determine layer type from config or other properties
        const layerType = layer.name.includes('conv2d') ? 'Conv2D' :
                          layer.name.includes('dense') ? 'Dense' :
                          layer.name.includes('flatten') ? 'Flatten' :
                          layer.name.includes('max_pooling2d') ? 'MaxPooling2D' : 'Unknown';

        switch(layerType) {
            case 'Dense':
                neuronCount = config.units;
                console.log('[layerType], config.units (neuronCount): ', layerType,neuronCount);
                console.log('outputShape: ', layer.outputShape);
                break;
            case 'Conv2D':
                // each neuron in this layer is a filter.
                // Each filter, in reality, is a group of neurons, with count being (kernelSize * kernelSize * inputChannels)
                // so the weights going into each of the neuron is the sum of the weights of input neurons
                neuronCount = config.filters;
                console.log('[layerType], config.filters (neuronCount): ', layerType,neuronCount);
                console.log('outputShape: ', layer.outputShape);
                break;
            case 'Flatten':
            case 'MaxPooling2D':
                // Assuming the last dimension of outputShape as neuron count, a rough approximation
                const outputShape = layer.outputShape;
                neuronCount = Array.isArray(outputShape) ? outputShape[outputShape.length - 1] : 0;
                console.log('[layerType], neuronCount: ', layerType, neuronCount);
                console.log('outputShape: ', layer.outputShape);
                break;
            default:
                console.log(`Layer type ${layerType} is not explicitly handled. Detected as: ${layerType}`);
                return;
        }

        // Call your function to create neuron representations for this layer
        createNeuronsForLayer(layerIndex, neuronCount, layerType);
        
    });
    console.log('Model visualization complete');
}

// chatgpt4 generated code
// CREATES nuerons as spheres
const flattenMat = new T.MeshBasicMaterial({color: 0x0000ff});
const denseMat = new T.MeshBasicMaterial({color: 0xff4500});
const conv2dMat = new T.MeshBasicMaterial({color: 0x00ff00});
const maxPoolingMat = new T.MeshBasicMaterial({color: 0xff00ff});
function createNeuronsForLayer(layerIndex, neuronCount,layerType) {
    const geometry = new T.SphereGeometry(1, 32, 32);
    for (let i = 0; i < neuronCount; i++) {
        let material;
        if(layerType === 'Flatten') material = flattenMat;
        else if(layerType === 'Dense') material = denseMat;
        else if(layerType === 'Conv2D') material = conv2dMat;
        else if(layerType === 'MaxPooling2D') material = maxPoolingMat;
        const sphere = new T.Mesh(geometry, material);
        sphere.position.x = -50 + i * 5; // Adjust positioning based on your layout needs
        sphere.position.y = 50 -layerIndex * 20; // Layer-based Y positioning
        sphere.userData = { layerIndex: layerIndex, neuronIndex: i, layerType: layerType }; // Store layer and neuron index
        nodes.add(sphere);
        scene.add(sphere);
        
    }
}

// next objective: visualize the kernels of conv2d layers.
// When hovered, display the weights of the neuron as a image. 
// The idea of connecting Conv2d layer via weights is not appplicable as the layer is not fully connected.
async function drawWeights(neuron, modelWeights) {
    // Clear previous lines
    scene.children
        .filter(child => child.userData.layerType === 'line')
        .forEach(child => scene.remove(child));

    const layerIndex = neuron.userData.layerIndex;
    const neuronIndex = neuron.userData.neuronIndex;
    let weights;
    var nextLayerNeuronCount;
    // ONLY the last two layers are fully connected with weights.
    // flatten layer technically can be connected from previous layer, but that would not be necessary as it is just a reshaping layer.
    if(neuron.userData.layerType === 'Flatten' || neuron.userData.layerType === 'Dense') {
        if(neuron.userData.layerType === 'Flatten') {
            weights = modelWeights[2].weights.slice(neuronIndex * 784, (neuronIndex + 1) * 784);
            nextLayerNeuronCount = modelWeights[2].shape[1];
        }
        else {
            weights = modelWeights[2].weights.slice(neuronIndex * 784, (neuronIndex + 1) * 784);
            nextLayerNeuronCount = modelWeights[2].shape[0];
        }
        for (let i = 0; i < nextLayerNeuronCount; i++) {
            var nextNeuron;
            if (neuron.userData.layerType === 'Dense')
                nextNeuron = scene.children.find(child => child.userData.layerIndex === layerIndex - 1 && child.userData.neuronIndex === i);
            else
                nextNeuron = scene.children.find(child => child.userData.layerIndex === layerIndex + 1 && child.userData.neuronIndex === i);
            if (!nextNeuron) {
                continue;
            }
            // T.Geometry is deprecated, use BufferGeometry
            let points = [];
            points.push(neuron.position);
            points.push(nextNeuron.position);
            let lineGeometry = new T.BufferGeometry().setFromPoints(points);
            // line color should be based on the weight value
            
            let color = new T.Color("red");
            if(weights[i] > 0)  {
                color = new T.Color("green");
            }
            else {
                color = new T.Color("red");
            }
            let lineMaterial = new T.LineBasicMaterial({color: color});
            let line = new T.Line(lineGeometry, lineMaterial);
            
            line.userData.layerType = 'line';
            scene.add(line);
        }
    }
    // if conv2d layer, display an 'image' made using the weights on a separate canvas
    else if(neuron.userData.layerType === 'Conv2D'){
        const container = document.getElementById('filtersContainer');
        container.innerHTML = '';
        let lN = neuron.userData.layerIndex;
        if(lN != 0) lN = 1;
        weights = modelWeights[lN].weights;
        console.log('-------------------');
        console.log('reading conv2d');
        console.log(modelWeights[lN].name);
        console.log(weights[0])
        /* display all filters on this layer
        for(let k=0; k<weights[0][0][0].length; k++) {
            const filterWeights = weights.map(row => row.map(col => col[0][k]));
            const normalizedFilter = normalizeFilter(filterWeights);
            displayFilter(normalizedFilter, `Filter ${k+1}`, 'filtersContainer');
        }
        */
        // display one filter (hovered neuron)
        const filterWeights = weights.map(row => row.map(col => col[0][neuronIndex]));
        const normalizedFilter = normalizeFilter(filterWeights);
        displayFilter(normalizedFilter, `Filter ${neuronIndex+1}`, 'filtersContainer');
    }
    
    

}

function normalizeFilter(filter) {
    const flatWeights = filter.flat();
    const max = Math.max(...flatWeights);
    const min = Math.min(...flatWeights);
    return filter.map(row => row.map(value => ((value-min)/(max-min)*255))); // normalized values to 0~255, representing pixel values
}

function displayFilter(filter, title, containerId) {
    const container = document.getElementById(containerId);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const scale = 10; // Scale up the filter for better visibility
    canvas.width = filter[0].length * scale;
    canvas.height = filter.length * scale;

    // Display title
    const filterTitle = document.createElement('div');
    filterTitle.textContent = title;
    container.appendChild(filterTitle);

    // Draw the filter
    filter.forEach((row, i) => {
        row.forEach((value, j) => {
            ctx.fillStyle = `rgb(${value},${value},${value})`; // Grayscale
            ctx.fillRect(j * scale, i * scale, scale, scale);
        });
    });
    container.appendChild(canvas);
}

const raycaster = new T.Raycaster();
const mouse = new T.Vector2();
let canHover = false;
window.addEventListener('click', event => {
    mouse.x = (event.clientX / 800) * 2 - 1;
    mouse.y = -(event.clientY / 800) * 2 + 1;
    canHover = !canHover;
});




let renderer = new T.WebGLRenderer();
renderer.setSize(800, 800);
document.body.style = 'display: flex; justify-content: left; align-items: center;';
document.body.appendChild(renderer.domElement);
let scene = new T.Scene();
let camera = new T.PerspectiveCamera(75, 800 / 800, 0.1, 1000);
camera.position.set(0, 0, 100);
scene.add(camera);


//let orbit = new OrbitControls(camera, renderer.domElement);

let container = document.createElement('div');
container.id = 'filtersContainer';
document.body.appendChild(container);
let model;
let modelWeights;
let nodes = new T.Group();
document.addEventListener('DOMContentLoaded', run);

async function run() {  
    /*
    const data = new MnistData();
    await data.load();
    await showExamples(data);

    const model = getModel();
    tfvis.show.modelSummary({name: 'Model Architecture'}, model);
    await train(model, data);
    */

    const data = new MnistData();
    await data.load();
    await showExamples(data);

    model = await tf.loadLayersModel('./my-model.json');
    tfvis.show.modelSummary({name: 'Model Architecture'}, model);
    modelWeights = await extractWeights(model);
    visualizeModelLayers(model);
    let drawing = false;
    createButtons(model, drawing);
    animate();
    doPrediction(model, data);
    showAccuracy(model, data);
    showConfusion(model, data);
}

function createButtons(model, drawing) {  
    // when hovered over a node, display the weights of the node
    // when inside the drawCanvas, let the user draw a number
    window.addEventListener('mousemove', event => {
        mouse.x = (event.clientX / 800) * 2 - 1;
        mouse.y = -(event.clientY / 800) * 2 + 1;
        // check if mouse is in the renderer. if so, enable raycast
        if(event.clientX < 800 && event.clientY < 800) {
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(scene.children);
            if (canHover && intersects.length > 0) {
                const neuron = intersects[0].object;
                if(!(neuron.userData.layerType === 'line'))
                    drawWeights(neuron, modelWeights);
            }
        }
    });
    let drawCanvas = document.createElement('canvas');
    drawCanvas.width = 280;
    drawCanvas.height = 280;
    let ctx = drawCanvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 280, 280);
    drawCanvas.style = 'margin: 4px; border: 5px solid orange;';
    drawCanvas.addEventListener('mousedown', () => drawing = true);
    drawCanvas.addEventListener('mouseup', () => drawing = false);
    drawCanvas.addEventListener('mouseleave', () => drawing = false);
    drawCanvas.addEventListener('mousemove', (event) => {
        console.log('drawing');
        if (drawing) {
            const rect = drawCanvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            draw(ctx, x, y);
        }
    });
    document.body.appendChild(drawCanvas);
    let clearButton = document.createElement('button');
    clearButton.textContent = 'Clear';
    clearButton.style = 'margin: 4px;';
    document.body.appendChild(clearButton);
    clearButton.addEventListener('click', () => {
        const ctx = drawCanvas.getContext('2d');
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, 280, 280);
    });

    let resizedCanvas = document.createElement('canvas');
    resizedCanvas.style = 'margin: 4px; border: 5px solid orange;';
    resizedCanvas.width = 28;
    resizedCanvas.height = 28;
    document.body.appendChild(resizedCanvas);
    let checkResizeButton = document.createElement('button');
    checkResizeButton.textContent = 'Check Resize';
    checkResizeButton.style = 'margin: 4px;';
    document.body.appendChild(checkResizeButton);
    checkResizeButton.addEventListener('click', () => {
        const imageData = ctx.getImageData(0, 0, 280, 280);
        // Resize the image to 28x28, then show it on the canvas
        const resized = tf.browser.fromPixels(imageData).resizeBilinear([28, 28]);
        tf.browser.toPixels(resized, resizedCanvas);
    });

    let checkNumberButton = document.createElement('button');
    checkNumberButton.textContent = 'Check Number';
    checkNumberButton.style = 'margin: 4px;';
    document.body.appendChild(checkNumberButton);
    checkNumberButton.addEventListener('click', () => {
        //const ctx = drawCanvas.getContext('2d');
        //const imageData = ctx.getImageData(0, 0, 280, 280);
        checkResizeButton.click();
        //const resized = tf.browser.fromPixels(drawCanvas).resizeBilinear([28, 28]).mean(2).expandDims(2).expandDims().toFloat();
        const imageData = ctx.getImageData(0, 0, 280, 280);
        const tensor = tf.browser.fromPixels(imageData, 1).resizeNearestNeighbor([28, 28]).expandDims(0).toFloat();
        const prediction = model.predict(tensor);
        //const preProcessed = resized.div(255.0); // Normalize pixel values to [0, 1]
        //const prediction = model.predict(preProcessed);
        const label = classNames[prediction.argMax(-1).dataSync()[0]];
        alert(`The number is: ${label}`);
    });

    let downloadButton = document.createElement('button');
    downloadButton.textContent = 'Download Model';
    downloadButton.style = 'margin: 4px;';
    document.body.appendChild(downloadButton);
    downloadButton.addEventListener('click', async () => {
        await model.save('downloads://my-model');
    });


}

function draw(ctx, x, y) {
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2); // Draw larger dots to match the canvas scale
    ctx.fill();
}


function animate() {
    // when clicked on a node, display the weights of the node
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
}


const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}

async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});

  labels.dispose();
}
