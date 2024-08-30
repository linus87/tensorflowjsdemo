// Tiny TFJS train / predict example.
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

function showImage(x: tf.Tensor4D, container: HTMLElement | null) {
  if (container == null) return;

  let canvas = document.createElement('canvas');
  canvas.width = x.shape[0];
  canvas.height = x.shape[1];
  container.appendChild(canvas);
  const y:tf.Tensor3D  = x.squeeze();
  const z:tf.Tensor3D = y.toInt();

  tf.browser.toPixels(z, canvas);
}

// Tiny TFJS train / predict example.
async function run() {

  let fromBlobImgElement = document.getElementById('fromBlobImg') as HTMLImageElement;
  const imageOrigialPixels = fromBlobImgElement != null && tf.browser.fromPixels(fromBlobImgElement) as tf.Tensor3D;

  if (imageOrigialPixels) {
    imageOrigialPixels.print();
    console.log(imageOrigialPixels);

    const layer = tf.layers.upSampling2d({size: [2, 2], interpolation: 'bilinear'});
    const result = layer.apply(imageOrigialPixels.expandDims()) as tf.Tensor4D;
    result.print();
    console.log(result);

    showImage(result, document.getElementById('filter-container'));
  }

}
  
run();