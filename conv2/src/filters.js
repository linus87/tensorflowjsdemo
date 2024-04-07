// create a tensor of shape [3, 3, 3, 32] with tf.tensor4d 
// and apply a convolutional layer with tf.conv2d.
// Calculate the total number of elements for the shape [3, 3, 3, 32]
const totalElements = 3 * 3 * 3 * 32;

// Create a flat array with the total number of elements
// For example, you can fill it with zeros or random values
const data = new Float32Array(totalElements).fill(0); // Replace with your data or use a different method to fill the array

// Create the 4D tensor with the specified shape
const filters = tf.tensor4d(data, [3, 3, 3, 32]);

filters.print(); // Print the tensor to the console