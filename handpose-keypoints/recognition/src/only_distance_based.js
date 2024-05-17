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
const wieghts = tf.tensor([[3.2218969 , -1.8663737, -3.477216 , -2.1979196, -9.871151, 0.2024691, 2.190202  , -0.1882586, 2.0197496 , 1.6175656 ],
  [-6.0136957, 5.0744133 , 2.5621598 , -8.4135132, 1.3143432, 0.240703 , -5.2118402, -1.3409088, 2.033278  , 1.5436258 ],
  [-3.1186938, -4.5643411, 5.1157742 , 2.8699007 , 1.5178241, 0.2127987, -4.0384626, 2.8297927 , -3.9003229, -3.4731734],
  [-0.6022994, -2.4009905, -4.028862 , 3.1515486 , 2.549145 , 3.3925426, -4.204442 , -1.5043688, -1.7274327, -2.157166 ],
  [-2.9048569, -0.9780643, -2.2000251, 2.5857329 , 2.4415894, 1.4200983, 5.928185  , -3.0472519, -0.9030867, -6.6117096]]
);
const bias = tf.tensor([4.1865149, 0.903378, -1.2850181, -0.3665743, -1.1681888, -5.2610803, 0.3457573, 0.7096063, -0.6336471, 2.7892659]);
model.add(tf.layers.dense({weights: [wieghts, bias], units: 10, activation: 'softmax'}));

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
  const six_landmarks_response = await fetch('/test_data/six/annotations.json');
  const seven_landmarks_response = await fetch('/test_data/seven/annotations.json');
  const eight_landmarks_response = await fetch('/test_data/eight/annotations.json');
  const nine_landmarks_response = await fetch('/test_data/nine/annotations.json');

  const fit_landmarks = await fit_landmarks_response.json();
  const one_landmarks = await one_landmarks_response.json();
  const two_landmarks = await two_landmarks_response.json();
  const three_landmarks = await three_landmarks_response.json();
  const four_landmarks = await four_landmarks_response.json();
  const five_landmarks = await five_landmarks_response.json();
  const six_landmarks = await six_landmarks_response.json();
  const seven_landmarks = await seven_landmarks_response.json();
  const eight_landmarks = await eight_landmarks_response.json();
  const nine_landmarks = await nine_landmarks_response.json();

  const fit_landmarks_dataset = tf.data.array(fit_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([0], 10) };});
  const one_landmarks_dataset = tf.data.array(one_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([1], 10) };});
  const two_landmarks_dataset = tf.data.array(two_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([2], 10) };});
  const three_landmarks_dataset = tf.data.array(three_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([3], 10) };});
  const four_landmarks_dataset = tf.data.array(four_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([4], 10) };});
  const five_landmarks_dataset = tf.data.array(five_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([5], 10) };});
  const six_landmarks_dataset = tf.data.array(six_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([6], 10) };});
  const seven_landmarks_dataset = tf.data.array(seven_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([7], 10) };});
  const eight_landmarks_dataset = tf.data.array(eight_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([8], 10) };});
  const nine_landmarks_dataset = tf.data.array(nine_landmarks).map(annotations => {return {xs: convertAnnotationsIntoDistanceFromPalm(annotations), ys: tf.oneHot([9], 10) };});

  const landmarksDataset = fit_landmarks_dataset.concatenate(one_landmarks_dataset).concatenate(two_landmarks_dataset)
    .concatenate(three_landmarks_dataset).concatenate(four_landmarks_dataset).concatenate(five_landmarks_dataset)
    .concatenate(six_landmarks_dataset).concatenate(seven_landmarks_dataset).concatenate(eight_landmarks_dataset).concatenate(nine_landmarks_dataset).batch(1);
  await landmarksDataset.forEachAsync(e => {e.ys.print(); e.xs.print()});


  const history = await model.fitDataset(landmarksDataset, {
    validationBatches: 1,
    epochs: 100, // Number of iterations over the entire dataset
    validationData: landmarksDataset,
  });

  tfvis.show.history({name: 'History'}, history, ['loss', 'acc']);
}

trainModel();
const classNames = ['Fit', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPredict() {

  // Make predictions (again, this is just an example, replace with your actual data)
  const value = JSON.parse(document.getElementById('number').value);
  console.log(value);
  const predictions = model.predict(tf.tensor([value], [1, 5, 1]) );
  predictions.print(); // This will output probabilities. You can threshold at 0.5 for binary classification.
}



