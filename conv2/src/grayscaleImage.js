async function grayscaleImage(imageElement) {
  // Load the image as a tensor
  const imageTensor = tf.browser.fromPixels(imageElement);

  // Define the weights for the RGB channels
  const weights = tf.tensor([0.299, 0.587, 0.114], [1, 1, 3, 1]);

  return await tf.conv2d(imageTensor, weights, 1, 'same').toInt();
}