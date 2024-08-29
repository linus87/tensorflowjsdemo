// Tiny TFJS train / predict example.
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

const imageSize = 224;

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

  const imageOrigialPixels = fromBlobImgElement != null && tf.browser.fromPixels(fromBlobImgElement) as tf.Tensor3D;

  if (imageOrigialPixels != false) {
    const filter = tf.randomNormal([3, 3, 3, 1]) as tf.Tensor4D;
    const conv2dResult = tf.conv2d(imageOrigialPixels.toFloat(), filter, 1, 'same').relu() as tf.Tensor3D;

    const heatmapValues = tf.transpose(conv2dResult.squeeze() as tf.Tensor2D);

    tfvis.render.heatmap(document.getElementById('heatmap-container') as HTMLElement, {values: heatmapValues}, {height: 250, width: 300});
    
    const conv2dTranspostResult = tf.conv2dTranspose(conv2dResult, filter, imageOrigialPixels.shape, 1, 'same').toInt();
    showImage(conv2dTranspostResult.clipByValue(0, 255) as tf.Tensor3D, document.getElementById('transpose-container'));
  }
}
  
run();