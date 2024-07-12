// Tiny TFJS train / predict example.
import * as tf from '@tensorflow/tfjs';

const imageSize = 200;
const outlineWidth = 1;
const colorShrehold = 10;
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
  const thresholdTensor = tf.fill([imageSize, imageSize, channelSize], colorShrehold);
  const result = tf.sub(maximumOutline, thresholdTensor).relu().toInt() as tf.Tensor3D;
  tf.browser.toPixels(result, resultImg);

  const mask = result.sign().toInt();

  const cropped = tf.mul(mask, compresedPixels).toInt() as tf.Tensor3D;
  tf.browser.toPixels(cropped, croppedImg);
}
  
run();