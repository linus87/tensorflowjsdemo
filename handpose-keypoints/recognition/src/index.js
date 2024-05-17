// Tiny TFJS train / predict example.
// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [5, 3], dtype: 'float32', batchSize: 1}));
model.add(tf.layers.flatten());
const wieghts = tf.tensor([[1.6987662 , -0.4296388, -0.5051717, 0.0030878 , -0.2143582, -0.9961827, 0.0605836 , 0.4643893 , 0.254651  , 0.8242526 ],
  [1.3168875 , -1.9477118, -2.3948805, 0.0151759 , -5.8560162, 1.1249017 , -0.3558149, 0.1986638 , 0.9865385 , 1.4572579 ],
  [1.0000079 , -0.229706 , -0.3569923, -2.5906696, -1.4802282, 0.1390387 , 0.1951267 , 0.2924493 , -0.0220554, 0.0121025 ],
  [-2.4300234, 0.1464898 , 0.5810929 , 0.3444096 , 0.5830713 , -0.0469978, -0.2218327, -0.991793 , -0.7248257, 2.0701017 ],
  [-3.9386246, 0.8613836 , 0.0587598 , -6.3234119, 0.6406921 , 0.6613733 , -4.1485648, 0.9473058 , 0.2562975 , 0.7773027 ],
  [-0.1347977, 0.3234328 , 0.0632806 , -0.3122408, -0.0602926, -1.3568834, -0.2456511, 0.7599182 , -0.5311821, 1.1089654 ],
  [-0.1776538, 0.423507  , 0.1211953 , 0.4310383 , 0.1807836 , -0.1096858, -2.3962519, -0.0886655, -0.3332754, -1.48935  ],
  [-1.4463131, -2.3227775, 2.8510637 , 1.7126267 , 0.8240985 , 0.0615713 , -1.9400361, 1.6379671 , -1.4328369, -0.4737009],
  [0.3257663 , 0.0016365 , -0.1810418, -0.0017515, 0.5003766 , -0.2938623, 0.8934789 , 0.9326625 , -0.5754777, -1.0703064],
  [-1.1986367, 0.3302342 , 0.4031355 , 0.8586873 , 0.5594596 , 0.1668479 , -1.5925847, -0.2758606, 0.3578517 , -2.979301 ],
  [0.0436408 , -0.9436289, -3.4885683, 1.9520931 , 1.6053344 , 2.2377837 , -0.8813921, -0.20859  , -0.8215124, 0.3913002 ],
  [0.7821718 , 0.2953174 , 0.5206591 , 0.1513676 , 0.1361032 , 0.4312325 , 0.7522117 , -1.3930432, 0.3245987 , -1.2119868],
  [-3.147495 , 0.2201311 , -0.0131794, 0.1709955 , 0.5932463 , 0.484091  , 1.3665788 , -1.2322593, 0.4393818 , -2.3546207],
  [-1.1441706, -0.553286 , 0.2305605 , 0.7441587 , 0.8329062 , 1.5144669 , 2.7023418 , -1.3941554, -2.1465931, -0.4134799],
  [0.605894  , -0.0765987, -0.7850617, -0.1496258, 0.2581504 , 0.1072229 , 0.6786585 , -0.5827736, -0.3676325, -0.8645644]]
);
const bias = tf.tensor([1.5626029, -0.0644332, -0.3157211, -0.3741303, 0.0797911, -1.2187728, -0.3272693, 0.1363837, -0.3545237, 0.7232698]
);
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

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

function convertVectorsIntoAngles(annotations) {
  const fingerTensors = convertAnnotationsIntoVector(annotations);
  fingerTensors.print();
  const fingerSegmentLengthTensors = tf.norm(fingerTensors, 2, 2, true);
  fingerSegmentLengthTensors.print();

  const fingerVectors = fingerTensors.arraySync();
  const fingerSegmentVectorsDot = fingerVectors.map(segments => segments.reduce((accumulator, currentValue, currentIndex, array) => {
    let previous = array[currentIndex - 1];
    if (currentIndex === 1) return [currentValue[0] * previous[0] + currentValue[1] * previous[1] + currentValue[2] * previous[2]];
    else return accumulator.concat(currentValue[0] * previous[0] + currentValue[1] * previous[1] + currentValue[2] * previous[2]);
  }));
  console.log(fingerSegmentVectorsDot);

  const fingerSegmentLengths = fingerSegmentLengthTensors.arraySync();
  const fingerSegmentsAngles = fingerSegmentVectorsDot.map((finger, fingerIndex) => finger.map((segment, segmentIndex) => segment / fingerSegmentLengths[fingerIndex][segmentIndex] / fingerSegmentLengths[fingerIndex][segmentIndex+1]) );
  console.log(fingerSegmentsAngles);
  
  return tf.tensor(fingerSegmentsAngles);
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

  const fit_landmarks_dataset = tf.data.array(fit_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([0], 10) };});
  const one_landmarks_dataset = tf.data.array(one_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([1], 10) };});
  const two_landmarks_dataset = tf.data.array(two_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([2], 10) };});
  const three_landmarks_dataset = tf.data.array(three_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([3], 10) };});
  const four_landmarks_dataset = tf.data.array(four_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([4], 10) };});
  const five_landmarks_dataset = tf.data.array(five_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([5], 10) };});
  const six_landmarks_dataset = tf.data.array(six_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([6], 10) };});
  const seven_landmarks_dataset = tf.data.array(seven_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([7], 10) };});
  const eight_landmarks_dataset = tf.data.array(eight_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([8], 10) };});
  const nine_landmarks_dataset = tf.data.array(nine_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([9], 10) };});

  const landmarksDataset = fit_landmarks_dataset.concatenate(one_landmarks_dataset).concatenate(two_landmarks_dataset)
    .concatenate(three_landmarks_dataset).concatenate(four_landmarks_dataset).concatenate(five_landmarks_dataset)
    .concatenate(six_landmarks_dataset).concatenate(seven_landmarks_dataset).concatenate(eight_landmarks_dataset).concatenate(nine_landmarks_dataset).batch(1);

  let handposesData = await landmarksDataset.toArray();
  console.log(handposesData);

  let xs = [];
  let ys = [];
  handposesData.forEach((e) => {
    xs.push(e.xs.squeeze());
    ys.push(e.ys.squeeze());
  });

  const x = tf.stack(xs);
  const y = tf.stack(ys);
  x.print();
  y.print();

  const history = await model.fit(x, y, {
    batchSize: 1,
    epochs: 500, // Number of iterations over the entire dataset,
    shuffle: true
  });

  tfvis.show.history({name: 'History'}, history, ['loss', 'acc']);
}

trainModel();