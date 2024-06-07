/**
 * 这个模型使用了两个特征：1）手指各个关节依序的角度；
 * 模型准确度在90%左右。
 */
const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [5, 3], dtype: 'float32', batchSize: 1}));
model.add(tf.layers.activation({activation: 'relu'}));
model.add(tf.layers.flatten());
const weights = tf.tensor([[2.6792464  , -0.1284471 , -1.2321272 , -0.1325671 , -0.390527 , -1.929329 , -0.3023147, -0.1430994, 0.5542675 , 0.9830784 ],
  [1.943956   , -1.7109621 , -4.800981  , 0.0976109  , -9.9609938, 0.6910396 , -0.237523 , 0.0280359 , 1.5288496 , 1.3840127 ],
  [1.9757785  , 0.3262194  , -0.5301979 , -4.0314174 , -2.9220436, -0.6023026, 0.1743883 , 0.1437453 , 0.1036535 , -0.2289325],
  [-3.7385347 , 1.1610054  , 1.274611   , 0.4821508  , 0.5695992 , -0.9761741, -0.4962827, -1.9281876, -1.1213689, 2.3291736 ],
  [-10.6518965, 1.7002691  , 0.0097694  , -13.7794876, 0.5282374 , -0.2293963, -9.2419062, 0.4681238 , 1.0593387 , -0.1498479],
  [-0.3700415 , 1.0328523  , -0.0720835 , -0.4102685 , -0.2485951, -2.4548948, -0.1454944, 0.529552  , -0.0440059, 0.3935931 ],
  [0.008165   , -0.9207053 , 0.5709046  , 0.9465309  , 0.2883216 , 0.8264157 , -2.9745202, 1.1924678 , -1.8175056, 0.0608441 ],
  [-5.0596676 , -10.3690672, 3.731184   , 3.1169398  , 0.8296307 , 0.0206157 , -3.3289559, 2.1287699 , -0.781601 , -2.1320622],
  [0.3943456  , -0.1442003 , 0.0736892  , 0.1064765  , 0.4020594 , -0.7696643, 1.2554613 , 1.4073056 , -0.7551255, -0.9852814],
  [-0.6690325 , -1.4801208 , 0.8355061  , 1.4966519  , 0.8827388 , 1.6156948 , -1.6998782, -0.690966 , 0.4433773 , -1.6289861],
  [-1.1204109 , -1.7191638 , -10.0524502, 3.4258187  , 2.521013  , 2.8622591 , -2.6692367, -1.2601445, 0.7389023 , -2.22525  ],
  [1.1477914  , 0.5003247  , 0.6155859  , 0.2856768  , 0.1138307 , 0.4654483 , 1.2318634 , -0.7427952, 0.0055097 , -1.7463526],
  [-3.50142   , -0.4043065 , 0.264992   , 0.6459005  , 1.4325125 , 2.3030064 , 3.0427976 , -2.6458461, 0.0520031 , -1.3088561],
  [-3.3043356 , 0.3887366  , 2.5457544  , 1.5184165  , 1.5967454 , 1.9459459 , 4.7628121 , -2.1613848, -1.8382939, -2.3669491],
  [1.4065731  , -0.3569722 , -1.0031796 , -0.075759  , 0.2443803 , -0.1743997, 1.3251401 , -0.5767801, 0.4922364 , -1.7802712]]
);
const bias = tf.tensor( [2.7544267, 0.3939092, -0.5001178, -0.5747058, -0.0161514, -2.4116037, -0.7003429, -0.1985726, -0.2718692, 0.7677516]
);
model.add(tf.layers.dense({weights: [weights, bias], units: 10, activation: 'softmax'}));

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

  const fist_landmarks_dataset = tf.data.array(fit_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([0], 10) };});
  const one_landmarks_dataset = tf.data.array(one_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([1], 10) };});
  const two_landmarks_dataset = tf.data.array(two_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([2], 10) };});
  const three_landmarks_dataset = tf.data.array(three_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([3], 10) };});
  const four_landmarks_dataset = tf.data.array(four_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([4], 10) };});
  const five_landmarks_dataset = tf.data.array(five_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([5], 10) };});
  const six_landmarks_dataset = tf.data.array(six_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([6], 10) };});
  const seven_landmarks_dataset = tf.data.array(seven_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([7], 10) };});
  const eight_landmarks_dataset = tf.data.array(eight_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([8], 10) };});
  const nine_landmarks_dataset = tf.data.array(nine_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([9], 10) };});

  const landmarksDataset = fist_landmarks_dataset.concatenate(one_landmarks_dataset).concatenate(two_landmarks_dataset)
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
    epochs: 200, // Number of iterations over the entire dataset,
    shuffle: true
  });

  tfvis.show.history({name: 'History'}, history, ['loss', 'acc']);
}

trainModel();