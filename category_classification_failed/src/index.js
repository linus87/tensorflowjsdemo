// Tiny TFJS train / predict example.
const numFeatures = 1;

// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.dense({units: 2, activation: 'relu', inputShape: [numFeatures]})); // numFeatures is the number of input features
model.add(tf.layers.dense({units: 2, activation: 'softmax'}));

// Compile the model with a binary loss function and an optimizer
model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

model.summary();
tfvis.show.modelSummary({name: 'Model Summary'}, model);

// Generate some synthetic data for training
const numbers = tf.range(0, 100, 1); // Generate numbers from 0 to 99
// const lables = numbers.mod(tf.scalar(2)).toInt();
const labels = numbers.mod(tf.scalar(2)).reshape([-1, 1]).toInt(); // 0 for even, 1 for odd numbers

numbers.print();
labels.print(true);

const classNames = ['Even', 'Odd'];
// Convert to one-hot encoding (even = [1, 0], odd = [0, 1])
const oneHotLabels = tf.oneHot(labels.squeeze(), 2);
oneHotLabels.print(true);

// Train the model
async function trainModel() {
  const history = await model.fit(numbers, oneHotLabels, {
    epochs: 100, // Number of iterations over the entire dataset,
    validationSplit: 0.2, // 50% of the data will be used for validation
    shuffle: true
  });

  tfvis.show.history({name: 'History'}, history, ['loss', 'acc']);
}

trainModel();

function doPredict() {
  // Make predictions (again, this is just an example, replace with your actual data)
  const value = Number(document.getElementById('number').value);
  const predictions = model.predict(tf.tensor2d([value], [1, 1]) );
  predictions.print(); // This will output probabilities. You can threshold at 0.5 for binary classification.

  predictions.argMax(1).data().then(index => {
    document.getElementById('prediction').value = classNames[index];
  });

  // To get binary labels (0 or 1), you can apply a threshold to the predictions
  const thresholdedPredictions = predictions.greater(tf.scalar(0.5)).cast('float32');
  thresholdedPredictions.print();
}

document.getElementById('predict').addEventListener('click', doPredict);

