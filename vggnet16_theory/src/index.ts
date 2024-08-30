// Tiny TFJS train / predict example.
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

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

  const filter1_1 = tf.variable(tf.randomNormal([3, 3, 3, 1])) as tf.Tensor4D;
  const filter1_2 = tf.variable(tf.randomNormal([3, 3, 1, 1])) as tf.Tensor4D;

  const conv1 = tf.conv2d(compresedPixels, filter1_1, [1, 1], 'same').relu();
  const conv1HeatMap = tf.transpose(conv1.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('featuremap-1') as HTMLElement, {values: conv1HeatMap}, {height: 250, width: 300});
  
  const conv2 = tf.conv2d(conv1 as tf.Tensor3D, filter1_2, [1, 1], 'same').relu();
  const conv2HeatMap = tf.transpose(conv2.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('featuremap-2') as HTMLElement, {values: conv2HeatMap}, {height: 250, width: 300});

  const pool1 = tf.pool(conv2 as tf.Tensor3D, [2, 2], 'max', 'same', [1, 1], 2);
  const pool1HeatMap = tf.transpose(pool1.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('pool-1') as HTMLElement, {values: pool1HeatMap}, {height: 250, width: 300});

  const filter2_1 = tf.variable(tf.randomNormal([3, 3, 1, 1])) as tf.Tensor4D;
  const filter2_2 = tf.variable(tf.randomNormal([3, 3, 1, 1])) as tf.Tensor4D;

  const conv2_1 = tf.conv2d(pool1, filter2_1, [1, 1], 'same').relu();
  const conv2_1HeatMap = tf.transpose(conv2_1.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('featuremap-2-1') as HTMLElement, {values: conv2_1HeatMap}, {height: 250, width: 300});
  
  const conv2_2 = tf.conv2d(conv2_1 as tf.Tensor3D, filter2_2, [1, 1], 'same').relu();
  const conv2_2HeatMap = tf.transpose(conv2_2.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('featuremap-2-2') as HTMLElement, {values: conv2_2HeatMap}, {height: 250, width: 300});

  const pool2 = tf.pool(conv2_2 as tf.Tensor3D, [2, 2], 'max', 'same', [1, 1], 2);
  const pool2HeatMap = tf.transpose(pool2.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('pool-2') as HTMLElement, {values: pool2HeatMap}, {height: 250, width: 300});

  const filter3_1 = tf.variable(tf.randomNormal([3, 3, 1, 1])) as tf.Tensor4D;
  const filter3_2 = tf.variable(tf.randomNormal([3, 3, 1, 1])) as tf.Tensor4D;
  const filter3_3 = tf.variable(tf.randomNormal([3, 3, 1, 1])) as tf.Tensor4D;

  const conv3_1 = tf.conv2d(pool2, filter3_1, [1, 1], 'same').relu();
  const conv3_1HeatMap = tf.transpose(conv3_1.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('featuremap-3-1') as HTMLElement, {values: conv3_1HeatMap}, {height: 250, width: 300});
  
  const conv3_2 = tf.conv2d(conv3_1 as tf.Tensor3D, filter3_2, [1, 1], 'same').relu();
  const conv3_2HeatMap = tf.transpose(conv3_2.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('featuremap-3-2') as HTMLElement, {values: conv3_2HeatMap}, {height: 250, width: 300});

  const conv3_3 = tf.conv2d(conv3_2 as tf.Tensor3D, filter3_3, [1, 1], 'same').relu();
  const conv3_3HeatMap = tf.transpose(conv3_3.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('featuremap-3-3') as HTMLElement, {values: conv3_3HeatMap}, {height: 250, width: 300});

  const pool3 = tf.pool(conv3_3 as tf.Tensor3D, [2, 2], 'max', 'same', [1, 1], 2);
  const pool3HeatMap = tf.transpose(pool3.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('pool-3') as HTMLElement, {values: pool3HeatMap}, {height: 250, width: 300});

  const filter4_1 = tf.variable(tf.randomNormal([3, 3, 1, 1])) as tf.Tensor4D;
  const filter4_2 = tf.variable(tf.randomNormal([3, 3, 1, 1])) as tf.Tensor4D;
  const filter4_3 = tf.variable(tf.randomNormal([3, 3, 1, 1])) as tf.Tensor4D;

  const conv4_1 = tf.conv2d(pool3, filter4_1, [1, 1], 'same').relu();
  const conv4_1HeatMap = tf.transpose(conv4_1.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('featuremap-4-1') as HTMLElement, {values: conv4_1HeatMap}, {height: 250, width: 300});
  
  const conv4_2 = tf.conv2d(conv4_1 as tf.Tensor3D, filter4_2, [1, 1], 'same').relu();
  const conv4_2HeatMap = tf.transpose(conv4_2.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('featuremap-4-2') as HTMLElement, {values: conv4_2HeatMap}, {height: 250, width: 300});

  const conv4_3 = tf.conv2d(conv4_2 as tf.Tensor3D, filter4_3, [1, 1], 'same').relu();
  const conv4_3HeatMap = tf.transpose(conv4_3.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('featuremap-4-3') as HTMLElement, {values: conv4_3HeatMap}, {height: 250, width: 300});

  const pool4 = tf.pool(conv4_3 as tf.Tensor3D, [2, 2], 'max', 'same', [1, 1], 2);
  const pool4HeatMap = tf.transpose(pool4.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('pool-4') as HTMLElement, {values: pool4HeatMap}, {height: 250, width: 300});

  const filter5_1 = tf.variable(tf.randomNormal([3, 3, 1, 1])) as tf.Tensor4D;
  const filter5_2 = tf.variable(tf.randomNormal([3, 3, 1, 1])) as tf.Tensor4D;
  const filter5_3 = tf.variable(tf.randomNormal([3, 3, 1, 1])) as tf.Tensor4D;

  const conv5_1 = tf.conv2d(pool4, filter5_1, [1, 1], 'same').relu();
  const conv5_1HeatMap = tf.transpose(conv5_1.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('featuremap-5-1') as HTMLElement, {values: conv5_1HeatMap}, {height: 250, width: 300});
  
  const conv5_2 = tf.conv2d(conv5_1 as tf.Tensor3D, filter5_2, [1, 1], 'same').relu();
  const conv5_2HeatMap = tf.transpose(conv5_2.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('featuremap-5-2') as HTMLElement, {values: conv5_2HeatMap}, {height: 250, width: 300});

  const conv5_3 = tf.conv2d(conv5_2 as tf.Tensor3D, filter5_3, [1, 1], 'same').relu();
  const conv5_3HeatMap = tf.transpose(conv5_3.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('featuremap-5-3') as HTMLElement, {values: conv5_3HeatMap}, {height: 250, width: 300});

  const pool5 = tf.pool(conv5_3 as tf.Tensor3D, [2, 2], 'max', 'same', [1, 1], 2);
  const pool5HeatMap = tf.transpose(pool5.squeeze() as tf.Tensor2D);
  tfvis.render.heatmap(document.getElementById('pool-5') as HTMLElement, {values: pool5HeatMap}, {height: 250, width: 300});

}
  
run();