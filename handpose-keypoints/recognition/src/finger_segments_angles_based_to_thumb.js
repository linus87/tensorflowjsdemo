/**
 * 这个模型使用了两个特征：1）手指各个关节与手指根部到手掌根部的向量的角度；2）手指第一个节点与拇指第一个节点的角度（分辨1和8的区别）。
 */
const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [5, 4], dtype: 'float32', batchSize: 1}));
// model.add(tf.layers.activation({activation: 'relu'}));
model.add(tf.layers.flatten());
const weights = tf.tensor([[0.0985235 , -0.6725039, -1.2263875, -0.0937763, 0.5963337 , -1.0649683, -0.3961747, -0.2379668, 0.2905979 , 0.0608547 ],
  [-1.3550634, -0.2655806, -3.1202619, 1.3800374 , -7.3700428, 0.9856706 , 0.2600968 , -1.0554063, 1.5129522 , -0.1480823],
  [-1.5781322, -2.2796509, -4.3344288, -2.42031  , -5.0786481, 2.7798607 , 1.4855461 , 0.8891832 , 3.1994624 , -1.5210412],
  [-0.0366645, -0.6371205, -0.9775266, -0.0926177, -3.3165455, 0.0347334 , -0.0983607, 0.9136122 , 0.9690129 , -0.1066127],
  [-1.4887283, 0.3831153 , 0.433804  , 0.1670349 , 0.0479938 , -1.0575091, -0.0960088, -1.5847441, -0.4718798, 1.0536159 ],
  [-9.7079906, 2.0049376 , 0.3559208 , -4.9729481, 0.9168058 , 0.2879395 , -3.8547606, -2.1058066, 0.4426471 , 0.4375353 ],
  [-5.7502975, 2.7114549 , 0.7774933 , -4.6634321, 0.8895071 , 0.6071057 , -3.3791742, -1.0654057, 1.1403334 , -3.4309611],
  [-0.3887093, 1.3558276 , -1.9523437, 0.2702711 , -1.787586 , 0.6456208 , -0.5642493, 1.9637153 , -3.2067657, 1.1430703 ],
  [0.2698068 , -0.7911412, 0.0506371 , 0.0793767 , 0.5682659 , 0.094432  , -0.2798071, 0.9040821 , -0.3197476, 0.7527086 ],
  [-2.0634866, -1.7396237, 2.7190077 , 1.5125903 , 0.7885121 , 1.0997355 , -0.8157103, 3.424114  , 0.6641881 , -1.0728474],
  [-0.7934334, -0.434527 , 3.1834834 , 1.4401653 , 0.7600469 , 1.4012491 , -1.2341585, 0.8431059 , 0.1802796 , -1.382588 ],
  [-1.1354349, -0.1803125, 0.0434342 , 0.3066014 , -0.838285 , 0.0248158 , -1.4463301, 2.7075381 , -2.9135816, 1.4200735 ],
  [1.4180015 , -1.1864495, 0.7817494 , 0.3491427 , 0.4054699 , 0.5627315 , 0.3232037 , -1.6530511, 0.9955652 , 0.561291  ],
  [-0.0981907, -0.1780868, -3.3392837, 2.2062073 , 1.4402421 , 2.1553123 , -0.4165385, 0.2327493 , -0.0115665, -0.2134603],
  [-1.1533828, 0.2030584 , -2.2000394, 1.2616612 , 1.2276801 , 2.7671597 , -1.4319299, -1.6073364, -0.401849 , 0.3427812 ],
  [-0.29368  , -0.6983641, 0.4207021 , -0.1197562, -0.4247456, 0.1004268 , -0.9685546, 2.0843844 , -1.7460412, 0.5301157 ],
  [-2.5319591, 1.5839123 , 0.8829629 , 1.0084035 , 0.6661327 , -0.1621108, 1.8903419 , -4.2818165, 0.2969776 , -0.9362968],
  [-1.3081889, 0.0037497 , 3.6093566 , 1.4722894 , 1.2488894 , 2.1109562 , 4.5408335 , -0.259241 , -0.629019 , -0.6904672],
  [-1.0627041, 0.5833057 , -1.8352082, 1.3834679 , 1.6975383 , 2.5533414 , 3.2776897 , -0.1557869, -1.6216304, -0.2878756],
  [-0.7649097, 0.2990404 , -0.0136587, -0.9813275, 0.4241155 , -0.5998385, -0.5051569, 0.8175391 , 0.569575  , -2.3579574]]
);
const bias = tf.tensor( [0.0538871, -0.2203505, -0.5859668, -0.5580947, 0.3606549, -0.9791526, -0.4906588, 0.2716785, 0.1118058, 0.1563296]);
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
    angles.push(dot / fiveFingerSegmentLengthArray[0][2] / fiveFingerSegmentLengthArray[i][1]);
  }
  
  return tf.tensor(angles);
}

function mergeFingerAnnotations(annotations) {
  const angles = convertVectorsIntoAnglesWithFirstSegment(annotations);
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

  const fist_landmarks_dataset = tf.data.array(fit_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([0], 10) };});
  const one_landmarks_dataset = tf.data.array(one_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([1], 10) };});
  const two_landmarks_dataset = tf.data.array(two_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([2], 10) };});
  const three_landmarks_dataset = tf.data.array(three_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([3], 10) };});
  const four_landmarks_dataset = tf.data.array(four_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([4], 10) };});
  const five_landmarks_dataset = tf.data.array(five_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([5], 10) };});
  const six_landmarks_dataset = tf.data.array(six_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([6], 10) };});
  const seven_landmarks_dataset = tf.data.array(seven_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([7], 10) };});
  const eight_landmarks_dataset = tf.data.array(eight_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([8], 10) };});
  const nine_landmarks_dataset = tf.data.array(nine_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([9], 10) };});

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