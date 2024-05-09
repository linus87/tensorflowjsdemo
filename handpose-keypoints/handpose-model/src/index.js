// Tiny TFJS train / predict example.
// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [5, 4, 3], dtype: 'float32', batchSize: 1}));
model.add(tf.layers.batchNormalization({axis: -1}));
// model.add(tf.layers.permute({dims:[2, 1]}));
// Flatten the output of the embedding layer to be able to connect it to a dense layer
// model.add(tf.layers.layerNormalization({axis: 0}));
// model.add(tf.layers.flatten());
model.add(tf.layers.conv2d({filters: 10, kernelSize:1, strides:1, padding:'same'}));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({units: 2, activation: 'softmax'}));

// Compile the model with a binary loss function and an optimizer
model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

model.summary();
tfvis.show.modelSummary({name: 'Model Summary'}, model);

function toTensorFromPalm(positions) {
  return positions.reduce((accumulator, currentValue, currentIndex, array) => {
    let previous = array[currentIndex - 1];
    if (currentIndex === 1) return [[currentValue[0] - previous[0], currentValue[1] - previous[1], currentValue[2] - previous[2]]];
    else return accumulator.concat([[currentValue[0] - previous[0], currentValue[1] - previous[1], currentValue[2] - previous[2]]]);
  })
}

function handleAnnotations(annotations) {
  let xs = [];
  const palmBase = annotations.palmBase;
  xs.push(toTensorFromPalm(palmBase.concat(annotations.thumb)));
  xs.push(toTensorFromPalm(palmBase.concat(annotations.indexFinger)));
  xs.push(toTensorFromPalm(palmBase.concat(annotations.middleFinger)));
  xs.push(toTensorFromPalm(palmBase.concat(annotations.ringFinger)));
  xs.push(toTensorFromPalm(palmBase.concat(annotations.pinky)));
  
  return tf.tensor(xs);
}

// Train the model
async function trainModel() {
    // Generate some synthetic data for training
  const five_landmarks_response = await fetch('/test_data/five/annotations.json');
  const one_landmarks_response = await fetch('/test_data/one/annotations.json');

  const five_landmarks = await five_landmarks_response.json();
  const one_landmarks = await one_landmarks_response.json();

  const five_landmarks_dataset = tf.data.array(five_landmarks).map(annotations => {
    return {xs: handleAnnotations(annotations), ys: tf.tensor([0])};
  });
  const one_landmarks_dataset = tf.data.array(one_landmarks).map(annotations => {return {xs: handleAnnotations(annotations), ys: tf.tensor([1])};});

  const landmarksDataset = five_landmarks_dataset.concatenate(one_landmarks_dataset).batch(1);

  const history = await model.fitDataset(landmarksDataset, {
    validationBatches: 1,
    epochs: 50, // Number of iterations over the entire dataset
    validationData: landmarksDataset,
  });

  tfvis.show.history({name: 'History'}, history, ['loss', 'acc']);
}

trainModel();
const classNames = ['Five', 'One'];

function doPredict() {

  // Make predictions (again, this is just an example, replace with your actual data)
  const value = JSON.parse(document.getElementById('number').value);
  console.log(value);
  const predictions = model.predict(tf.tensor([value], [1, 21, 3]) );
  predictions.print(); // This will output probabilities. You can threshold at 0.5 for binary classification.

  predictions.argMax(1).data().then(index => {
    document.getElementById('prediction').value = classNames[index];
  });

  // To get binary labels (0 or 1), you can apply a threshold to the predictions
  const thresholdedPredictions = predictions.greater(tf.scalar(0.5)).cast('float32');
  thresholdedPredictions.print();
}

document.getElementById('predict').addEventListener('click', doPredict);

