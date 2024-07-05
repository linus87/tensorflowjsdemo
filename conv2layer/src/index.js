// Tiny TFJS train / predict example.


import loadImage from "./loadImage";
import * as tf from '@tensorflow/tfjs';

const imageSize = 200;
const kernelSize = 3;
const filterSize = 32;

// Tiny TFJS train / predict example.
let model;
async function run() {
  // const imageBlob = await loadImage('/cat.jpeg');
  // console.log(imageBlob) ;
  let fromBlobImgElement = document.getElementById('fromBlobImg');
  // fromBlobImgElement.src = URL.createObjectURL(imageBlob);

  const imageOrigialPixels = tf.browser.fromPixels(fromBlobImgElement);
  const compresedPixels = tf.image.resizeBilinear(imageOrigialPixels, [imageSize, imageSize]).toInt();
  tf.browser.toPixels(compresedPixels, document.getElementById('compressedImg'));

  // Create a simple model.
  model = tf.sequential();
  
  model.add(tf.layers.inputLayer({batchInputShape: [1, null, null, 3]})); 
  // model.add(tf.layers.resizing({height: imageSize, width: imageSize, interpolation: 'bilinear',  cropToAspectRatio: true})); 
  model.add(tf.layers.centerCrop({height: imageSize, width: imageSize})); 
  model.add(tf.layers.conv2d({filters:filterSize, kernelSize:kernelSize, strides:1, padding:'same', dataFormat:'channelsLast', activation: 'relu'}));
  // model.add(tf.layers.dense({units: 3, kernelInitializer:"ones", useBias: true}));

  model.summary();

  const result = model.predict(compresedPixels.reshape([1, imageSize, imageSize, 3]));
  const filterResults = tf.split(result, filterSize, 3);

  filterResults[0].print();

  for (let i = 0; i < filterSize; i++) {
    let canvas = document.createElement('canvas');
    canvas.width = imageSize;
    canvas.height = imageSize;
    document.getElementById('filter-container').appendChild(canvas);
    tf.browser.toPixels(filterResults[i].squeeze().toInt(), canvas);
  }
  console.log(model.getLayer("conv2d_Conv2D1").getWeights());
  
}
  
run();