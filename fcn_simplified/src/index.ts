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

  // Create a simple model.
  const model = tf.sequential();
  
  model.add(tf.layers.inputLayer({batchInputShape: [1, null, null, channelSize]})); 
  model.add(tf.layers.conv2d({filters: 64, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));
  model.add(tf.layers.conv2d({filters: 64, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));

  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: 2, padding: 'same'}));
  model.add(tf.layers.conv2d({filters: 128, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));
  model.add(tf.layers.conv2d({filters: 128, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));

  model.add(tf.layers.conv2d({filters: 1024, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));
  model.add(tf.layers.conv2d({filters: 1024, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));

  model.add(tf.layers.conv2dTranspose({filters: categorySize, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu'}));
  model.add(tf.layers.upSampling2d({size: [2, 2]}));

  model.summary();


  // let result = model.predict(compresedPixels.reshape([1, imageSize, imageSize, channelSize])) as tf.Tensor;
  if (imageOrigialPixels != false) {
    // let result = model.predict(imageOrigialPixels.reshape([1, imageSize, imageSize, channelSize])) as tf.Tensor;
    let result = model.predict(imageOrigialPixels.expandDims()) as tf.Tensor;
    result = result.squeeze();
    result = result.argMax(2);
    console.log(result);
    result.print();
    // const {values, indices} = result.unique(2);

    const colorFilter = tf.rand([1, 1, 1, 3], ()=> Math.floor(Math.random() * 255 / categorySize)) as tf.Tensor4D;

    result = tf.conv2d(result.expandDims(2).toFloat() as tf.Tensor3D, colorFilter, 1, 'same').toInt();

    //  result = tf.conv2d(result.reshape([imageSize, imageSize, 1]).toFloat() as tf.Tensor3D, [[[[100/21, 200/21, 255/21]]]], 1, 'same').toInt();

    // result.print();
    tf.browser.toPixels(result as tf.Tensor2D, resultImg);
  }
  
  

}
  
run();