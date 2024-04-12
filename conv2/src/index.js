// Tiny TFJS train / predict example.
async function run() {
  const imageElement = document.getElementById('fromBlobImg');
  const imageTensor = tf.browser.fromPixels(imageElement);

  const imageNetSizeImg = tf.image.resizeBilinear(imageTensor, [224, 224]);
  // const rgbTensor = imageTensor.toFloat().div(tf.scalar(255));
  tf.browser.toPixels(imageNetSizeImg.toInt(), document.getElementById('tf-resized-canvas'));

  // Define the shape of the filter
  const filter_shape = [3, 3, 3, 1];

  const filter = tf.variable(tf.randomNormal(filter_shape));
  filter.print();

  // stride [1,1] 
  const stride1Conv = tf.conv2d(imageNetSizeImg, filter, [1, 1], 'same').toInt();
  const stride1Values = await stride1Conv.array();

  // Render to visor
  const surface1 = { name: 'stride [1,1]', tab: 'Charts' };
  tfvis.render.heatmap(surface1, {values:stride1Values});

  // stride [2,2]
  const stride2Conv = tf.conv2d(imageNetSizeImg, filter, [2, 2], 'same').toInt();
  const stride2Values = await stride2Conv.array();

  // Render to visor
  const surface2 = { name: 'stride [2,2]', tab: 'Charts' };
  tfvis.render.heatmap(surface2, {values: stride2Values});

  // grayscale and stride [1,1]
  const grayscaleImageFilterShape = [3, 3, 1, 1];

  const grayscaleFilter = tf.variable(tf.randomNormal(grayscaleImageFilterShape));

  const tfGrayScaleImageTensor = tf.image.rgbToGrayscale(imageNetSizeImg);
  const stride3Conv = tf.conv2d(tfGrayScaleImageTensor, grayscaleFilter, [1, 1], 'same').toInt();
  const stride3Values = await stride3Conv.array();

  // Render to visor
  const surface3 = { name: 'after grayscale with stride [1,1]', tab: 'Charts' };
  tfvis.render.heatmap(surface3, {values: stride3Values});
}
  
run();