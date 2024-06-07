/**
 * 这个模型使用了两个特征：1）手指各个关节与手指根部到手掌根部的向量的角度；2）手指第一个节点与拇指第一个节点的角度（分辨1和8的区别）。
 */
const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [5, 3], dtype: 'float32', batchSize: 1}));
// model.add(tf.layers.activation({activation: 'relu'}));
model.add(tf.layers.flatten());
const weights = tf.tensor([[2.7869899  , -0.4658208, -2.610481 , -1.0986463 , -0.8682361 , -1.5621271, -2.2197423, 1.3461505 , -0.9030719, 0.8419355  ],
  [-0.1236348 , -0.8627891, -6.6246667, 1.8989277  , -12.4215593, 6.3173466 , -0.5811712, -1.0035888, 2.3939149 , 0.1221828  ],
  [-3.591758  , -2.8642085, -7.5537333, -8.8704243 , -12.6752825, -5.5704532, 1.2655401 , 2.3539646 , 4.1332049 , -3.9069471 ],
  [1.9363668  , 0.8400565 , -0.8623614, -0.7802492 , -0.3142502 , -0.9421286, -0.6564267, -1.2098279, -2.1461713, 3.0435143  ],
  [-19.8145218, 2.4813643 , 0.3028463 , -10.6608486, 0.3558249  , -0.470119 , -8.3541756, -3.2107887, 0.3795781 , 0.5360875  ],
  [-9.8751898 , 3.4934897 , 0.394973  , -10.4043674, -0.3070939 , -0.7589771, -4.9695506, -2.9040599, 0.9386003 , -6.0543213 ],
  [-0.6745908 , -0.4492018, 0.3881192 , 0.3163049  , -1.0516304 , 0.3638183 , -0.5534506, 3.6319494 , -4.3021827, 1.1606207  ],
  [-5.9742713 , -9.7683935, 4.1796789 , 1.2056311  , -0.1368611 , 0.3847823 , -3.4380915, 1.5904858 , -1.5640836, -8.7456398 ],
  [3.6518943  , -8.7689161, 4.8650279 , 1.4162582  , -0.6360688 , 0.5669364 , -4.0446692, -1.617806 , -1.1281683, -10.6607533],
  [2.5468507  , -1.2433568, 0.924372  , 1.4066391  , -0.4750915 , 0.417779  , 0.6023163 , -2.8475988, 1.1301413 , 1.4776584  ],
  [-4.9276958 , -2.0430412, -6.6868615, 1.5826559  , -0.2140248 , 0.489098  , -2.9625297, -2.4304652, -4.1845379, -1.2068957 ],
  [2.2311985  , -6.2537966, -6.6770005, 0.9857266  , 0.1530069  , 1.1668237 , -4.8191328, -10.119091, -3.1946909, 0.1924647  ],
  [-2.8613811 , 2.2647374 , 0.5102214 , 1.2446648  , -0.066806  , -0.392227 , 4.0642152 , -6.2494311, 1.7656797 , -0.9093608 ],
  [-6.1326084 , -0.2966537, 1.4791892 , 1.6981467  , 0.3339298  , 0.1744684 , 5.7388396 , -4.4922771, -7.0203414, -6.2441726 ],
  [-2.1875873 , 3.3550365 , -7.0402927, 1.7988775  , 0.4953503  , -0.0175977, 5.1878519 , -8.5789099, -9.8432703, -0.9701751 ]]
);
const bias = tf.tensor( [3.5148931, -0.3447134, -1.8910071, -2.2250586, -0.7316685, -0.9859235, -2.1752069, 2.0694568, -1.9049413, 0.0713049]);
model.add(tf.layers.dense({weights: [weights, bias], units: 10, activation: 'softmax'}));
// model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

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

function convertVectorsIntoAnglesWithFirstSegment(annotations) {
  const fingerTensors = convertAnnotationsIntoVector(annotations);
  fingerTensors.print();
  const fingerSegmentLengthTensors = tf.norm(fingerTensors, 2, 2, true);
  fingerSegmentLengthTensors.print();

  const fingerVectors = fingerTensors.arraySync();
  const fingerSegmentVectorsDot = fingerVectors.map(segments => segments.reduce((accumulator, currentValue, currentIndex, array) => {
    let previous = array[0];
    if (currentIndex === 1) return [currentValue[0] * previous[0] + currentValue[1] * previous[1] + currentValue[2] * previous[2]];
    else return accumulator.concat(currentValue[0] * previous[0] + currentValue[1] * previous[1] + currentValue[2] * previous[2]);
  }));

  const fingerSegmentLengths = fingerSegmentLengthTensors.arraySync();
  const fingerSegmentsAngles = fingerSegmentVectorsDot.map((fingerSegments, fingerIndex) => fingerSegments.map((segment, segmentIndex) => segment / fingerSegmentLengths[fingerIndex][0] / fingerSegmentLengths[fingerIndex][segmentIndex + 1]) );
  console.log(fingerSegmentsAngles);
  
  return tf.tensor(fingerSegmentsAngles);
}

function convertVectorsIntoAnglesToThumb(annotations) {
  const fiveFingersTensors = convertAnnotationsIntoVector(annotations);
  // fingerTensors.print();
  const fiveFingesrSegmentLengthTensors = tf.norm(fiveFingersTensors, 2, 2, true);
  const fiveFingerSegmentLengthArray = fiveFingesrSegmentLengthTensors.arraySync();
  // fingerSegmentLengthTensors.print();

  const fiveFingerVectors = fiveFingersTensors.arraySync();
  const base = fiveFingerVectors[0][2];
  let angles = [];
  for (let i=0; i<5; i++) {
    const segment = fiveFingerVectors[i][1];
    const dot = base[0] * segment[0] + base[1] * segment[1] + base[2] * segment[2];
    angles.push(dot / fiveFingerSegmentLengthArray[0][1] / fiveFingerSegmentLengthArray[i][1]);
  }
  
  return tf.tensor(angles);
}

function mergeFingerAnnotations(annotations) {
  const angles = convertVectorsIntoAngles(annotations);
  // angles.print();
  const distances = convertVectorsIntoAnglesToThumb(annotations).reshape([5, 1]);
  // distances.print();
  angles.concat(distances, 1).print();
  return angles.concat(distances, 1);
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

  const fist_landmarks_dataset = tf.data.array(fit_landmarks).map(annotations => {return {xs: convertVectorsIntoAnglesWithFirstSegment(annotations), ys: tf.oneHot([0], 10) };});
  const one_landmarks_dataset = tf.data.array(one_landmarks).map(annotations => {return {xs: convertVectorsIntoAnglesWithFirstSegment(annotations), ys: tf.oneHot([1], 10) };});
  const two_landmarks_dataset = tf.data.array(two_landmarks).map(annotations => {return {xs: convertVectorsIntoAnglesWithFirstSegment(annotations), ys: tf.oneHot([2], 10) };});
  const three_landmarks_dataset = tf.data.array(three_landmarks).map(annotations => {return {xs: convertVectorsIntoAnglesWithFirstSegment(annotations), ys: tf.oneHot([3], 10) };});
  const four_landmarks_dataset = tf.data.array(four_landmarks).map(annotations => {return {xs: convertVectorsIntoAnglesWithFirstSegment(annotations), ys: tf.oneHot([4], 10) };});
  const five_landmarks_dataset = tf.data.array(five_landmarks).map(annotations => {return {xs: convertVectorsIntoAnglesWithFirstSegment(annotations), ys: tf.oneHot([5], 10) };});
  const six_landmarks_dataset = tf.data.array(six_landmarks).map(annotations => {return {xs: convertVectorsIntoAnglesWithFirstSegment(annotations), ys: tf.oneHot([6], 10) };});
  const seven_landmarks_dataset = tf.data.array(seven_landmarks).map(annotations => {return {xs: convertVectorsIntoAnglesWithFirstSegment(annotations), ys: tf.oneHot([7], 10) };});
  const eight_landmarks_dataset = tf.data.array(eight_landmarks).map(annotations => {return {xs: convertVectorsIntoAnglesWithFirstSegment(annotations), ys: tf.oneHot([8], 10) };});
  const nine_landmarks_dataset = tf.data.array(nine_landmarks).map(annotations => {return {xs: convertVectorsIntoAnglesWithFirstSegment(annotations), ys: tf.oneHot([9], 10) };});

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
    epochs: 500, // Number of iterations over the entire dataset,
    shuffle: true
  });

  tfvis.show.history({name: 'History'}, history, ['loss', 'acc']);
}

trainModel();