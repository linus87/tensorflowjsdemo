// Tiny TFJS train / predict example.
import * as tf from '@tensorflow/tfjs';

const imageSize = 224;
const channelSize = 3;
const redSensedDistance = 8;
const greenSensedDistance = 8;
const blueSensedDistance = 8;

const edgeThreshold = 50;

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

function sensedImage(image: tf.Tensor3D, rDistance: number, gDistance: number, bDistance: number) {
  const [width, height] = image.shape;
  const rMod = tf.fill([width, height, 1], rDistance);
  const gMod = tf.fill([width, height, 1], gDistance);
  const bMod = tf.fill([width, height, 1], bDistance);
  const allMod = tf.concat([rMod, gMod, bMod], 2);

  return image.sub(image.mod(allMod));
}

function seekEdge(image: tf.Tensor3D) {
  const [width, height] = image.shape;
  const padded = tf.pad(image, [[1,1],[1,1],[0,0]], 255);

  const topRegion = padded.slice([0, 1], [width, height]);
  const leftRegion = padded.slice([1, 0], [width, height]);
  const bottomRegion = padded.slice([2, 1], [width, height]);
  const rightRegion = padded.slice([1, 2], [width, height]);

  const top = tf.max(tf.sub(image, topRegion).abs(), 2);
  const right = tf.max(tf.sub(image, leftRegion).abs(), 2);
  const left = tf.max(tf.sub(image, rightRegion).abs(), 2);
  const bottom = tf.max(tf.sub(image, bottomRegion).abs(), 2);

  return tf.max(tf.stack([top, right, left, bottom], 2), 2);

}

function showEdge(edge: tf.Tensor3D, threshold: number, container: HTMLElement | null) {
  if (container == null) return;

  const [width, height] = edge.shape;
  const thresholdMatrix = tf.fill([width, height], threshold);

  const backgroundTensor = tf.fill([width, height, 3], 255);
  const frontendTensor = tf.fill([width, height, 3], 0);

  const edges = tf.where(edge.less(thresholdMatrix).expandDims(2), backgroundTensor, frontendTensor);
  showImage(edges, container);
  return edges;
}

function catchImg(name: String, redSensedDistance: number, greenSensedDistance: number, blueSensedDistance: number, edgeThreshold: number) {
  let fromBlobImgElement = document.getElementById(name + '-img') as HTMLImageElement;
  const imageOrigialPixels: tf.Tensor3D = fromBlobImgElement != null && tf.browser.fromPixels(fromBlobImgElement);
  const [width, height, channels] = imageOrigialPixels.shape;

  const sensed = sensedImage(imageOrigialPixels, redSensedDistance, greenSensedDistance, blueSensedDistance);
  showImage(sensed, document.getElementById(name + '-sensed-image'));

  const edge = seekEdge(sensed);
  edge.print();

  const blackWhiteImg = showEdge(edge, edgeThreshold, document.getElementById(name + '-edged-image'));
  blackWhiteImg.print();
}

// Tiny TFJS train / predict example.
async function run() {

  catchImg("cat", redSensedDistance, greenSensedDistance, blueSensedDistance, edgeThreshold);
  catchImg("nba", redSensedDistance, greenSensedDistance, blueSensedDistance, edgeThreshold);
  catchImg("people", 2, 2, 2, 25);
}

run();