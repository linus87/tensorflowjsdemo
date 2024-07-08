// Tiny TFJS train / predict example.
import * as tf from '@tensorflow/tfjs';

const imageSize = 200;
const outlineWidth = 2;
const colorShrehold = 20;
const channelSize = 3;

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

// Tiny TFJS train / predict example.
let model;
async function run() {

  let fromBlobImgElement = document.getElementById('fromBlobImg') as HTMLImageElement;
  let container: HTMLElement | null = document.getElementById('filter-container');
  let outlineContainer: HTMLElement | null = document.getElementById('outline-container');
  let compressedImg = document.getElementById('compressedImg') as HTMLCanvasElement || undefined;
  let resultImg = document.getElementById('result') as HTMLCanvasElement || undefined;

  const imageOrigialPixels = fromBlobImgElement != null && tf.browser.fromPixels(fromBlobImgElement);
  const compresedPixels = tf.image.resizeBilinear(imageOrigialPixels, [imageSize, imageSize]).toInt();
  tf.browser.toPixels(compresedPixels, compressedImg);

  const upper = tf.pad(compresedPixels, [[outlineWidth, 0], [0, 0], [0, 0]]).slice([0, 0, 0], [imageSize, imageSize, channelSize]);
  const right = tf.pad(compresedPixels, [[0, 0], [0,outlineWidth], [0, 0]]).slice([0, outlineWidth, 0], [imageSize, imageSize, channelSize]);
  const bottom = tf.pad(compresedPixels, [[0, outlineWidth], [0, 0], [0, 0]]).slice([outlineWidth, 0, 0], [imageSize, imageSize, channelSize]);
  const left = tf.pad(compresedPixels, [[0, 0], [outlineWidth, 0], [0, 0]]).slice([0, 0, 0], [imageSize, imageSize, channelSize]);

  showImage(upper, container);
  showImage(right, container);
  showImage(bottom, container);
  showImage(left, container);

  const upperOutline = tf.sub(upper, compresedPixels).abs().toInt() as tf.Tensor3D;
  showImage(upperOutline, outlineContainer);

  const rightOutline = tf.sub(right, compresedPixels).abs().toInt() as tf.Tensor3D;
  showImage(rightOutline, outlineContainer);

  const bottomOutline = tf.sub(bottom, compresedPixels).abs().toInt() as tf.Tensor3D;
  showImage(bottomOutline, outlineContainer);

  const leftOutline = tf.sub(left, compresedPixels).abs().toInt() as tf.Tensor3D;
  showImage(leftOutline, outlineContainer);

  const maximumOutline = upperOutline.maximum(rightOutline).maximum(bottomOutline).maximum(leftOutline) as tf.Tensor3D;
  const result = tf.clipByValue(maximumOutline, colorShrehold, 255).toInt();
  tf.browser.toPixels(result, resultImg);

  // tf.addN([upperOutline, rightOutline, bottomOutline, leftOutline]);


  // const all = tf.pad(compresedPixels, [[paddingSize, paddingSize], [paddingSize, paddingSize], [0, 0]]);
  // showImage(all, container);

  // const batch = compresedPixels.reshape([1, imageSize, imageSize, channelSize]) as tf.Tensor4D;

  // const result = tf.image.cropAndResize(batch, 
  //   [[0, paddingSize, imageSize, imageSize + paddingSize], 
  //   [paddingSize, paddingSize * 2, imageSize + paddingSize, imageSize + paddingSize * 2], 
  //   [paddingSize * 2, paddingSize, imageSize + paddingSize * 2, imageSize + paddingSize], 
  //   [paddingSize, 0, imageSize + paddingSize, imageSize]], [0, 1, 0, 0], [imageSize, imageSize]);

  // const cropped = tf.split(result, 4, 0);

  // for (let i = 0; i < 4; i++) {
  //   let img = cropped[i].squeeze().toInt() as tf.Tensor3D;
  //   img.print();
  //   showImage(img, container);
  // }
  // console.log(result);



  // let rightCanvas = document.createElement('canvas');
  // canvas.width = imageSize;
  // canvas.height = imageSize;
  // document.getElementById('filter-container').appendChild(canvas);
  // tf.browser.toPixels(right.squeeze().toInt(), rightCanvas);

  // let bottomCanvas = document.createElement('canvas');
  // canvas.width = imageSize;
  // canvas.height = imageSize;
  // document.getElementById('filter-container').appendChild(canvas);
  // tf.browser.toPixels(bottom.squeeze().toInt(), bottomCanvas);

  // let leftCanvas = document.createElement('canvas');
  // canvas.width = imageSize;
  // canvas.height = imageSize;
  // document.getElementById('filter-container').appendChild(canvas);
  // tf.browser.toPixels(left.squeeze().toInt(), leftCanvas);

  // // Create a simple model.
  // model = tf.sequential();
  
  // model.add(tf.layers.inputLayer({batchInputShape: [1, null, null, channelSize]})); 
  // model.add(tf.layers.zeroPadding2d({padding: [[2, 2], [2, 2]], dataFormat: 'channelsLast'})); 
  // model.add(tf.layers.centerCrop({height: imageSize, width: imageSize})); 
  // // model.add(tf.layers.depthwiseConv2d({depthMultiplier:1, kernelSize:kernelSize, strides:2, padding:'same', dataFormat:'channelsLast', activation: 'relu'}));

  // model.summary();

  // const result = model.predict(compresedPixels.reshape([1, imageSize, imageSize, channelSize]));
  // let resultCanvasElement = document.getElementById('result') as HTMLCanvasElement;
  // tf.browser.toPixels(result.squeeze().toInt(), resultCanvasElement);
  // const filterResults = tf.split(result, channelSize, 3);

  // for (let i = 0; i < channelSize; i++) {
  //   let canvas = document.createElement('canvas');
  //   canvas.width = imageSize;
  //   canvas.height = imageSize;
  //   document.getElementById('filter-container').appendChild(canvas);
  //   tf.browser.toPixels(filterResults[i].squeeze().toInt(), canvas);
  // }
  // console.log(model.getLayer("depthwise_conv2d_DepthwiseC").getWeights());
  
}
  
run();