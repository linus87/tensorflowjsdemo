// Tiny TFJS train / predict example.
const numFeatures = 1;

// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [5], dtype: 'int32', batchSize: 1}));
model.add(tf.layers.dense({units: 6, activation: 'softmax'}));

// Compile the model with a binary loss function and an optimizer
model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

model.summary();
tfvis.show.modelSummary({name: 'Model Summary'}, model);

// Generate some synthetic data for training
const fingers = tf.tensor([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]], [6, 5], 'int32');
// const lables = numbers.mod(tf.scalar(2)).toInt();
const labels = tf.tensor([0, 1, 2, 3, 4, 5], [6], 'int32'); // 0 for even, 1 for odd numbers

labels.print(true);

const classNames = ['Fist', 'One', 'Two', 'Three', 'Four', 'Five'];
// Convert to one-hot encoding (even = [1, 0], odd = [0, 1])
const oneHotLabels = tf.oneHot(labels, 6);
oneHotLabels.print(true);

// Train the model
async function trainModel() {
  const history = await model.fit(fingers, oneHotLabels, {
    epochs: 8000, // Number of iterations over the entire dataset,
    validationSplit: 1, // 50% of the data will be used for validation
    shuffle: true
  });

  tfvis.show.history({name: 'History'}, history, ['loss', 'acc']);
}

trainModel();

function doPredict() {
  // Make predictions (again, this is just an example, replace with your actual data)
  const value = Number(document.getElementById('number').value);
  const predictions = model.predict(tf.tensor2d([value], [5]) );
  predictions.print(); // This will output probabilities. You can threshold at 0.5 for binary classification.

  predictions.argMax(1).data().then(index => {
    document.getElementById('prediction').value = classNames[index];
  });

  // To get binary labels (0 or 1), you can apply a threshold to the predictions
  const thresholdedPredictions = predictions.greater(tf.scalar(0.5)).cast('float32');
  thresholdedPredictions.print();
}

document.getElementById('predict').addEventListener('click', doPredict);

