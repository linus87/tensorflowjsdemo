// Tiny TFJS train / predict example.
import * as tf from '@tensorflow/tfjs';

const imageSize = 224;
const channelSize = 3;
const categorySize = 21;

function showImage(x: tf.Tensor3D, container: HTMLElement | null) {
  if (container == null) return;

  let canvas = document.createElement('canvas');
  canvas.width = imageSize;
  canvas.height = imageSize;
  container.appendChild(canvas);
  const y:tf.Tensor3D  = x.squeeze();
  const z:tf.Tensor3D = y.toInt();
  tf.browser.toPixels(z, canvas);
}

const colors = tf.rand([categorySize, channelSize], ()=> Math.floor(Math.random() * 255 / categorySize), 'int32');

// Tiny TFJS train / predict example.
async function run() {

  let fromBlobImgElement = document.getElementById('fromBlobImg') as HTMLImageElement;
  let container: HTMLElement | null = document.getElementById('filter-container');
  let outlineContainer: HTMLElement | null = document.getElementById('outline-container');
  let compressedImg = document.getElementById('compressedImg') as HTMLCanvasElement || undefined;
  let resultImg = document.getElementById('result') as HTMLCanvasElement || undefined;
  let croppedImg = document.getElementById('cropped') as HTMLCanvasElement || undefined;

  const imageOrigialPixels = fromBlobImgElement != null && tf.browser.fromPixels(fromBlobImgElement);
  const compresedPixels = tf.image.resizeBilinear(imageOrigialPixels, [imageSize, imageSize]).toInt();
  tf.browser.toPixels(compresedPixels, compressedImg);

  // Create a simple model.
  const model = tf.sequential();

  model.add(tf.layers.inputLayer({batchInputShape: [1, imageSize, imageSize, channelSize]})); 
  model.add(tf.layers.conv2d({filters: 64, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));
  model.add(tf.layers.conv2d({filters: 64, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));

  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: 2, padding: 'same'}));
  model.add(tf.layers.conv2d({filters: 128, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));
  model.add(tf.layers.conv2d({filters: 128, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));

  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: 2, padding: 'same'}));
  model.add(tf.layers.conv2d({filters: 256, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));
  model.add(tf.layers.conv2d({filters: 256, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));
  model.add(tf.layers.conv2d({filters: 256, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));

  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: 2, padding: 'same'}));
  model.add(tf.layers.conv2d({filters: 512, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));
  model.add(tf.layers.conv2d({filters: 512, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));
  model.add(tf.layers.conv2d({filters: 512, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));

  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: 2, padding: 'same'}));
  model.add(tf.layers.conv2d({filters: 512, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));
  model.add(tf.layers.conv2d({filters: 512, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));
  model.add(tf.layers.conv2d({filters: 512, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));

  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: 2, padding: 'same'}));

  // model.add(tf.layers.conv2d({filters: 4096, kernelSize: 7, strides: 7, padding: 'same', activation: 'relu'}));
  // model.add(tf.layers.conv2d({filters: 4096, kernelSize: 1, strides: 1, padding: 'same', activation: 'relu'}));

  // model.add(tf.layers.conv2d({filters: 1000, kernelSize: 1, strides: 1, padding: 'same', activation: 'softmax'}));

  model.add(tf.layers.flatten());

  model.add(tf.layers.dense({units: 4096, activation: 'relu'}));
  model.add(tf.layers.dense({units: 4096, activation: 'relu'}));

  model.add(tf.layers.dense({units: 1000, activation: 'softmax'}));

  model.summary();

  const result = model.predict(compresedPixels.reshape([1, imageSize, imageSize, channelSize])) as tf.Tensor;
  result.print();
  tf.sum(result).print();
}
  
run();