// Tiny TFJS train / predict example.
// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [5, 1], dtype: 'float32', batchSize: 1}));
// model.add(tf.layers.conv1d({filters: 5, kernelSize: 1, strides: 1, activation: 'sigmoid'}));
// model.add(tf.layers.globalMaxPooling1d());
// model.add(tf.layers.permute({dims:[2, 1]}));
// Flatten the output of the embedding layer to be able to connect it to a dense layer
// model.add(tf.layers.layerNormalization({axis: 0}));
// model.add(tf.layers.flatten());
// model.add(tf.layers.conv2d({filters: 10, kernelSize:1, strides:1, padding:'same'}));
model.add(tf.layers.flatten());
// model.add(tf.layers.layerNormalization({axis: -1}));
// model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
const wieghts = tf.tensor([[0.4285955 , -0.4196281, -2.3498352, -4.0158319, -8.9092369, 6.6489644 ],
  [-6.1820574, 9.5308514 , 4.6734819 , -8.0183411, 9.0317717 , 1.7292143 ],
  [-3.2929487, -8.0325375, 8.8502007 , 4.6598635 , -2.5088193, -1.5274763],
  [-1.7355452, -2.8372762, -5.6640782, 3.2940524 , 5.0955324 , 1.7308692 ],
  [-2.4063451, -1.977569 , -5.6118159, 3.7133763 , 4.4479179 , 0.9957244 ]]
);
const bias = tf.tensor([9.1208906, 1.0854325, -2.4544923, -2.1743815, -8.057126, -6.4409657]);
model.add(tf.layers.dense({weights: [wieghts, bias], units: 6, activation: 'softmax'}));

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

function distanceToPalm(palmBase, positions) {
  return positions.map(tensor => {
    const x = tensor[0] - palmBase[0];
    const y = tensor[1] - palmBase[1];
    const z = tensor[2] - palmBase[2];
    return Math.sqrt(x*x + y*y + z*z);
   });
}

function convertAnnotationsIntoDistanceFromPalm(annotations) {
  let xs = [];
  const palmBase = annotations.palmBase[0];
  xs.push(distanceToPalm(palmBase, annotations.thumb));
  xs.push(distanceToPalm(palmBase, annotations.indexFinger));
  xs.push(distanceToPalm(palmBase, annotations.middleFinger));
  xs.push(distanceToPalm(palmBase, annotations.ringFinger));
  xs.push(distanceToPalm(palmBase, annotations.pinky));

  let fingerTensors = [];
  fingerTensors.push(toTensorFromPalm(annotations.palmBase.concat(annotations.thumb)));
  fingerTensors.push(toTensorFromPalm(annotations.palmBase.concat(annotations.indexFinger)));
  fingerTensors.push(toTensorFromPalm(annotations.palmBase.concat(annotations.middleFinger)));
  fingerTensors.push(toTensorFromPalm(annotations.palmBase.concat(annotations.ringFinger)));
  fingerTensors.push(toTensorFromPalm(annotations.palmBase.concat(annotations.pinky)));

  let fingerLenghs = [];
  fingerTensors.forEach(fingerTensor => {
    let length = 0;
    
    fingerTensor.forEach(tensor => {
      length += Math.sqrt(tensor[0]*tensor[0] + tensor[1]*tensor[1] + tensor[2]*tensor[2]);
    });
    fingerLenghs.push(length);
  });

  xs = xs.map((tensor, index) => [tensor[3] / fingerLenghs[index]]);
  
  return tf.tensor(xs);
}

function convertAnnotationsIntoVector(annotations) {
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
  const fit_landmarks_response = await fetch('/test_data/fit/annotations.json');
  const one_landmarks_response = await fetch('/test_data/one/annotations.json');
  const two_landmarks_response = await fetch('/test_data/two/annotations.json');
  const three_landmarks_response = await fetch('/test_data/three/annotations.json');
  const four_landmarks_response = await fetch('/test_data/four/annotations.json');
  const five_landmarks_response = await fetch('/test_data/five/annotations.json');

  const fit_landmarks = await fit_landmarks_response.json();
  const one_landmarks = await one_landmarks_response.json();
  const two_landmarks = await two_landmarks_response.json();
  const three_landmarks = await three_landmarks_response.json();
  const four_landmarks = await four_landmarks_response.json();
  const five_landmarks = await five_landmarks_response.json();

  // const fit_landmarks_dataset = tf.data.array(fit_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.tensor([0, 0, 0, 0, 0])};});
  // const one_landmarks_dataset = tf.data.array(one_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.tensor([0, 1, 0, 0, 0])};});
  // const two_landmarks_dataset = tf.data.array(two_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.tensor([0, 1, 1, 0, 0])};});
  // const three_landmarks_dataset = tf.data.array(three_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.tensor([0, 0, 1, 1, 1])};});
  // const four_landmarks_dataset = tf.data.array(four_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.tensor([0, 1, 1, 1, 1])};});
  // const five_landmarks_dataset = tf.data.array(five_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.tensor([1, 1, 1, 1, 1])};});

  const fit_landmarks_dataset = tf.data.array(fit_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([0], 6) };});
  const one_landmarks_dataset = tf.data.array(one_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([1], 6) };});
  const two_landmarks_dataset = tf.data.array(two_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([2], 6) };});
  const three_landmarks_dataset = tf.data.array(three_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([3], 6) };});
  const four_landmarks_dataset = tf.data.array(four_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([4], 6) };});
  const five_landmarks_dataset = tf.data.array(five_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([5], 6) };});

  const landmarksDataset = fit_landmarks_dataset.concatenate(one_landmarks_dataset).concatenate(two_landmarks_dataset)
    .concatenate(three_landmarks_dataset).concatenate(four_landmarks_dataset).concatenate(five_landmarks_dataset).batch(1);
  await landmarksDataset.forEachAsync(e => {e.ys.print(); e.xs.print()});


  const history = await model.fitDataset(landmarksDataset, {
    validationBatches: 1,
    epochs: 100, // Number of iterations over the entire dataset
    validationData: landmarksDataset,
  });

  tfvis.show.history({name: 'History'}, history, ['loss', 'acc']);
}

trainModel();
const classNames = ['Fit', 'One', 'Two', 'Three', 'Four', 'Five'];

function doPredict() {

  // Make predictions (again, this is just an example, replace with your actual data)
  const value = JSON.parse(document.getElementById('number').value);
  console.log(value);
  const predictions = model.predict(tf.tensor([value], [1, 5, 1]) );
  predictions.print(); // This will output probabilities. You can threshold at 0.5 for binary classification.
}

document.getElementById('predict').addEventListener('click', doPredict);

