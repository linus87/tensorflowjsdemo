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

function sensedImage(image: tf.Tensor3D, rDistance: number, gDistance: number, bDistance: number) {
  const [width, height, channels] = image.shape;
  const rMod = tf.fill([width, height, 1], rDistance);
  const gMod = tf.fill([width, height, 1], gDistance);
  const bMod = tf.fill([width, height, 1], bDistance);
  const allMod = tf.concat([rMod, gMod, bMod], 2);

  return image.sub(image.mod(allMod));
}

// Tiny TFJS train / predict example.
async function run() {

  let fromBlobImgElement = document.getElementById('fromBlobImg') as HTMLImageElement;
  const imageOrigialPixels: tf.Tensor3D = fromBlobImgElement != null && tf.browser.fromPixels(fromBlobImgElement);
  const [width, height, channels] = imageOrigialPixels.shape;
  const imagePixelsVector = imageOrigialPixels.reshape([width * height, 3]);

  console.log(imagePixelsVector);

  const sensed = sensedImage(imageOrigialPixels, 128, 128, 128);
  sensed.print();
  showImage(sensed, document.getElementById('sensed-image'));
  // const differences = pixcelDifference(imagePixelsVector);
  // console.log(differences);
}
  
run();