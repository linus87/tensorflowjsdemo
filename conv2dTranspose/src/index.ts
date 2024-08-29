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

// Tiny TFJS train / predict example.
async function run() {

  let fromBlobImgElement = document.getElementById('fromBlobImg') as HTMLImageElement;
  let resultImg = document.getElementById('result') as HTMLCanvasElement || undefined;

  const imageOrigialPixels = fromBlobImgElement != null && tf.browser.fromPixels(fromBlobImgElement) as tf.Tensor3D;
  const compresedPixels = tf.image.resizeBilinear(imageOrigialPixels, [imageSize, imageSize]);

  const filter1 = tf.variable(tf.randomNormal([3, 3, 3, 3])) as tf.Tensor4D;
  const filter2 = tf.variable(tf.randomNormal([3, 3, 3, 1])) as tf.Tensor4D;

  const conv1 = tf.conv2d(compresedPixels, filter1, [1, 1], 'same').relu();
  showImage(conv1.toInt().clipByValue(0, 255) as tf.Tensor3D, document.getElementById('filter-container'));
  
  const conv2 = tf.conv2d(conv1 as tf.Tensor3D, filter2, [1, 1], 'same').relu();
  showImage(conv2.toInt().clipByValue(0, 255) as tf.Tensor3D, document.getElementById('filter-container'));

  const pool1 = tf.pool(conv2 as tf.Tensor3D, [2, 2], 'max', 'same', [1, 1], 2);
  showImage(pool1.toInt().clipByValue(0, 255) as tf.Tensor3D, document.getElementById('filter-container'));

  const conv2dT1 = tf.conv2dTranspose(pool1 as tf.Tensor3D, filter2, [224, 224, 3], 2, 'same');
  showImage(conv2dT1.toInt().clipByValue(0, 255) as tf.Tensor3D, document.getElementById('transpose-container'));

  const conv2dT2 = tf.conv2dTranspose(conv2dT1 as tf.Tensor3D, filter1, [224, 224, 3], 1, 'same');
  showImage(conv2dT2.toInt().clipByValue(0, 255) as tf.Tensor3D, document.getElementById('transpose-container'));

}
  
run();