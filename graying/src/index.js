// Tiny TFJS train / predict example.
async function grayscaleImage(imageElement) {
  // Load the image as a tensor
  const imageTensor = tf.browser.fromPixels(imageElement);

  // Define the weights for the RGB channels
  const weights = tf.tensor([0.299, 0.587, 0.114], [1, 1, 3, 1]);

  return await tf.conv2d(imageTensor, weights, 1, 'same').toInt();
}

// Example usage:
// Assuming you have an image element in your HTML with the id 'myImage'
const imageElement = document.getElementById('fromBlobImg');

const imageTensor = tf.browser.fromPixels(imageElement);
const tfGrayScaleImageTensor = tf.image.rgbToGrayscale(imageTensor);
tf.browser.toPixels(tfGrayScaleImageTensor, document.getElementById('tf-grayscale-canvas'));

// Call the function and process the result
grayscaleImage(imageElement).then(grayscaleTensor => {
//   // Now you can use the grayscaleTensor, for example, to display it in a canvas
  tf.browser.toPixels(grayscaleTensor, document.getElementById('grayscale-canvas'));
});