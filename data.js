/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export class MnistData {
    constructor() {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
    }

    async load() {
    // Make a request for the MNIST sprited image.
        const img = new Image();
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        const imgRequest = new Promise((resolve, reject) => {
            img.crossOrigin = '';
            img.onload = () => {
            img.width = img.naturalWidth;
            img.height = img.naturalHeight;

            // float32 array, each element is 4 bytes.
            // we have 65000 images, each image is 28*28=784 pixels (each pixel is a float32 number)
            // total of 65000 * 784 * 4 bytes
            const datasetBytesBuffer =
                new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

            const chunkSize = 5000;
            canvas.width = img.width;
            canvas.height = chunkSize;
            // total of 13 'chunks', each chuck storing 5000 images (65000/5000) 
            // so each chunk is 5000*784*4 bytes ( = 15.3MB )
            for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
                // Float32Array (arrayBuffer, offset(start_index), length)
                // changing float32array[i] will change arrayBuffer[i+offset], and i < length
                // this is because float32array will point to the memory location of arrayBuffer, not copy the values
                const datasetBytesView = new Float32Array(
                    datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
                    IMAGE_SIZE * chunkSize);
                ctx.drawImage(
                    img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
                    chunkSize);

                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                // imageData.data is a Uint8ClampedArray containing the image data
                // in RGBA order, where each color component is an integer between 0 and 255.
                // so to consider each pixel, iterate over imageData.data.length/4
                for (let j = 0; j < imageData.data.length / 4; j++) {
                // All channels hold an equal value since the image is grayscale, so
                // just read the red channel.
                // doing  +1 will read the green channel, +2 will read the blue channel, +3 will read the alpha channel
                datasetBytesView[j] = imageData.data[j * 4] / 255;
                // datasetBytesView points to the same memory as datasetBytesBuffer,
                // so assigning values (changing values) will change the values that are stored in where datasetBytesBuffer points to
                // '=' operator goes all the way to the memory location, and changes the value in that memory location
                // memory location is fixed, but the value in that memory location can be changed
                }
                
            }
            // so every 728 elements represent a single image, and each element is a pixel value
            this.datasetImages = new Float32Array(datasetBytesBuffer);
            
            resolve();
            };
            img.src = MNIST_IMAGES_SPRITE_PATH;
        });
        const labelsRequest = fetch(MNIST_LABELS_PATH);
        // await Promise.all([...]) will wait for all promises in [] to completely finish (either resolve or reject)
        const [imgResponse, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);
        this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

        // Create shuffled indices into the train/test set for when we select a
        // random dataset element for training / validation.
        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

        // Slice the the images and labels into train and test sets.
        this.trainImages =
            this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        this.trainLabels =
            this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
        this.testLabels =
            this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    }

    nextTrainBatch(batchSize) {
        return this.nextBatch(
            batchSize, [this.trainImages, this.trainLabels], () => {
                // shuffledTrainIndex is initialized by constructor to 0
                this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length;
                // trainIndices is initialized by load() to a shuffled array of indices
                return this.trainIndices[this.shuffledTrainIndex];
            }
        );
    }

    nextTestBatch(batchSize) {
        return this.nextBatch(
            batchSize, [this.testImages, this.testLabels], () => {
                // shuffledTestIndex is initialized by constructor to 0
                this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length;
                // testIndices is initialized by load() to a shuffled array of indices
                return this.testIndices[this.shuffledTestIndex];
            }
        );
    }

  // index input is defined as a function by nextTestBatch and nextTrainBatch
    nextBatch(batchSize, data, index) {
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

        for (let i = 0; i < batchSize; i++) {
            const idx = index();

            const image =
                data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
            batchImagesArray.set(image, i * IMAGE_SIZE);

            const label =
                data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
            batchLabelsArray.set(label, i * NUM_CLASSES);
        }

        const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
        const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

        return {xs, labels};
    }
}