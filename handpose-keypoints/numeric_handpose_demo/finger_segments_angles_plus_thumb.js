/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as handpose from '@tensorflow-models/handpose';
import * as tf from '@tensorflow/tfjs';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-webgl';

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
  // fingerTensors.print();
  const fingerSegmentLengthTensors = tf.norm(fingerTensors, 2, 2, true);
  // fingerSegmentLengthTensors.print();

  const fingerVectors = fingerTensors.arraySync();
  const fingerSegmentVectorsDot = fingerVectors.map(segments => segments.reduce((accumulator, currentValue, currentIndex, array) => {
    let previous = array[currentIndex - 1];
    if (currentIndex === 1) return [currentValue[0] * previous[0] + currentValue[1] * previous[1] + currentValue[2] * previous[2]];
    else return accumulator.concat(currentValue[0] * previous[0] + currentValue[1] * previous[1] + currentValue[2] * previous[2]);
  }));
  // console.log(fingerSegmentVectorsDot);

  const fingerSegmentLengths = fingerSegmentLengthTensors.arraySync();
  const fingerSegmentsAngles = fingerSegmentVectorsDot.map((finger, fingerIndex) => finger.map((segment, segmentIndex) => segment / fingerSegmentLengths[fingerIndex][segmentIndex] / fingerSegmentLengths[fingerIndex][segmentIndex+1]) );
  // console.log(fingerSegmentsAngles);
  
  return tf.tensor(fingerSegmentsAngles);
}

function convertVectorsIntoAnglesWithThumb(annotations) {
  const fiveFingersTensors = convertAnnotationsIntoVector(annotations);
  // fingerTensors.print();
  const fiveFingesrSegmentLengthTensors = tf.norm(fiveFingersTensors, 2, 2, true);
  const fiveFingerSegmentLengthArray = fiveFingesrSegmentLengthTensors.arraySync();
  // fingerSegmentLengthTensors.print();

  const fiveFingerVectors = fiveFingersTensors.arraySync();
  const base = fiveFingerVectors[0][2];
  let angles = [0];
  for (let i=1; i<5; i++) {
    const segment = fiveFingerVectors[i][1];
    const dot = base[0] * segment[0] + base[1] * segment[1] + base[2] * segment[2];
    angles.push(dot / fiveFingerSegmentLengthArray[0][1] / fiveFingerSegmentLengthArray[i][1]);
  }
  
  return tf.tensor(angles);
}

function mergeFingerAnnotations(annotations) {
  const angles = convertVectorsIntoAngles(annotations);
  // angles.print();
  const distances = convertVectorsIntoAnglesWithThumb(annotations).reshape([5, 1]);
  // distances.print();
  angles.concat(distances, 1).print();
  return angles.concat(distances, 1);
}

const classNames = ['fist', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
console.log(tf);
const numericHanposeModel = tf.sequential();
numericHanposeModel.add(tf.layers.inputLayer({inputShape: [5, 4], dtype: 'float32', batchSize: 1}));
// numericHanposeModel.add(tf.layers.activation({activation: 'relu'}));
numericHanposeModel.add(tf.layers.flatten());
const weights = tf.tensor( [[0.2270513 , -0.6192091, -0.9511225, -0.1353957, 0.4297972 , -0.7415566, -0.3694827, -0.0854145, 0.1118057 , 0.1150665 ],
  [-1.0104111, -0.193035 , -2.5700748, 1.1977096 , -6.0897903, 0.9177452 , 0.1805995 , -0.5874088, 1.1397069 , 0.042719  ],
  [-1.2235571, -1.9534203, -3.3800313, -2.391479 , -3.924088 , 2.4526556 , 1.2128221 , 1.0296872 , 2.698946  , -1.3142709],
  [0.0361734 , -0.5554109, -0.7451601, -0.166867 , -2.677882 , -0.0416087, -0.1335741, 0.9267353 , 0.8475127 , -0.0993176],
  [-1.3188541, 0.3422776 , 0.4440582 , 0.1180617 , -0.1019931, -0.7973424, -0.1750825, -1.2884477, -0.3874767, 0.9751761 ],
  [-7.4992323, 1.7849967 , 0.244619  , -4.4468012, 0.7280357 , 0.4381722 , -3.1865771, -1.6007805, 0.4412597 , 0.6198131 ],
  [-4.6372733, 2.3623414 , 0.574541  , -4.1000853, 0.6878875 , 0.709769  , -2.7625458, -1.1266997, 1.1050032 , -2.9177589],
  [-0.2423544, 1.0877806 , -1.5689397, 0.2292445 , -1.3210369, 0.5204976 , -0.5562656, 1.7228289 , -2.4514296, 0.9781774 ],
  [-0.014148 , -0.5241346, 0.0894252 , -0.011889 , 0.4619496 , 0.1981597 , -0.3756601, 0.7233446 , -0.3817016, 0.5523407 ],
  [-1.6126028, -1.5515256, 2.2276638 , 1.3347068 , 0.6760751 , 0.9261401 , -0.6547093, 2.8720222 , 0.5413029 , -1.0727036],
  [-0.5223942, -0.4839309, 2.4178016 , 1.3046114 , 0.6441616 , 1.1174117 , -1.0480083, 0.8051968 , 0.1334501 , -1.2832655],
  [-0.8070116, -0.0754418, 0.0344118 , 0.2654953 , -0.4946656, -0.0589004, -1.3746201, 2.2007852 , -2.497962 , 1.0810595 ],
  [0.9840119 , -0.7860935, 0.6891673 , 0.2395919 , 0.2841176 , 0.5980461 , 0.1767488 , -1.2904558, 0.7459002 , 0.1973339 ],
  [-0.1499208, -0.2057968, -2.3812072, 2.0208714 , 1.2090175 , 1.7486225 , -0.2911908, -0.1133822, 0.0764553 , -0.1927613],
  [-0.7636109, 0.138958  , -1.6780963, 1.1537224 , 0.995188  , 2.1840508 , -1.2207882, -1.2953069, -0.3232149, 0.2787057 ],
  [-0.0861528, -0.5052052, 0.3954813 , -0.1204592, -0.2027201, 0.0401821 , -0.9297804, 1.7623308 , -1.6368213, 0.2642547 ],
  [-2.0053275, 1.0708224 , 0.6930144 , 0.883886  , 0.4619488 , -0.1340366, 1.6337714 , -3.6122701, 0.3265196 , -0.5146638],
  [-1.2209759, -0.0622024, 2.4866185 , 1.2712262 , 0.940508  , 1.7066609 , 3.9031    , -0.6515406, -0.4757656, -0.6372826],
  [-0.7586719, 0.4722587 , -1.5812488, 1.2609439 , 1.3894348 , 1.9680492 , 2.8215344 , -0.1456861, -1.4974387, -0.2552495],
  [-0.4867397, 0.2190988 , 0.0840219 , -0.8891736, 0.4672077 , -0.6017702, -0.4475523, 0.7354755 , 0.3925236 , -1.8710792]]
);
const bias = tf.tensor([0.105031, -0.1566283, -0.4111324, -0.5677534, 0.2537085, -0.7283104, -0.5070054, 0.2456434, 0.023925, 0.1657945]);
numericHanposeModel.add(tf.layers.dense({weights: [weights, bias], units: 10, activation: 'softmax'}));

// Compile the model with a binary loss function and an optimizer
numericHanposeModel.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

/****** Segment line ******/

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}
tfjsWasm.setWasmPaths({
  'tfjs-backend-wasm.wasm': `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/tfjs-backend-wasm.wasm`,
  'tfjs-backend-wasm-simd.wasm': `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/tfjs-backend-wasm-simd.wasm`,
  'tfjs-backend-wasm-threaded-simd.wasm': `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/tfjs-backend-wasm-threaded-simd.wasm`,
});
let videoWidth, videoHeight, rafID, ctx, canvas, ANCHOR_POINTS,
  scatterGLHasInitialized = false, scatterGL, fingerLookupIndices = {
    thumb: [0, 1, 2, 3, 4],
    indexFinger: [0, 5, 6, 7, 8],
    middleFinger: [0, 9, 10, 11, 12],
    ringFinger: [0, 13, 14, 15, 16],
    pinky: [0, 17, 18, 19, 20]
  };  // for rendering each finger as a polyline

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 500;
const mobile = isMobile();
// Don't render the point cloud on mobile in order to maximize performance and
// to avoid crowding limited screen space.
const renderPointcloud = mobile === false;

const state = {
  backend: 'webgl',
  renderPointcloud: renderPointcloud
};

function drawPoint(y, x, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
}

function drawKeypoints(keypoints) {
  const keypointsArray = keypoints;

  for (let i = 0; i < keypointsArray.length; i++) {
    const y = keypointsArray[i][0];
    const x = keypointsArray[i][1];
    drawPoint(x - 2, y - 2, 3);
  }

  const fingers = Object.keys(fingerLookupIndices);
  for (let i = 0; i < fingers.length; i++) {
    const finger = fingers[i];
    const points = fingerLookupIndices[finger].map(idx => keypoints[idx]);
    drawPath(points, false);
  }
}

function drawPath(points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

let model;

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      width: mobile ? undefined : VIDEO_WIDTH,
      height: mobile ? undefined : VIDEO_HEIGHT
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();
  return video;
}

let video;
async function main() {
  await tf.setBackend(state.backend);
  if (!tf.env().getAsync('WASM_HAS_SIMD_SUPPORT') && state.backend == "wasm") {
    console.warn("The backend is set to WebAssembly and SIMD support is turned off.\nThis could bottleneck your performance greatly, thus to prevent this enable SIMD Support in chrome://flags");
  }
  model = await handpose.load();

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = e.message;
    info.style.display = 'block';
    throw e;
  }

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, videoWidth, videoHeight);
  ctx.strokeStyle = 'red';
  ctx.fillStyle = 'red';

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  // These anchor points allow the hand pointcloud to resize according to its
  // position in the input.
  // ANCHOR_POINTS = [
  //   [0, 0, 0], [0, -VIDEO_HEIGHT, 0], [-VIDEO_WIDTH, 0, 0],
  //   [-VIDEO_WIDTH, -VIDEO_HEIGHT, 0]
  // ];

  landmarksRealTime(video);
}

const handposeImg = document.getElementById('current-pose');

const landmarksRealTime = async (video) => {
  async function frameLandmarks() {

    ctx.drawImage(
      video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width,
      canvas.height);
    const predictions = await model.estimateHands(video);
    // console.log(predictions);
    if (predictions.length > 0) {

      const handposePredicts = numericHanposeModel.predict(mergeFingerAnnotations(predictions[0].annotations).reshape([1, 5, 4]));

      handposePredicts.argMax(1).data().then(index => {
          console.log(`${classNames[index]}`);
          handposeImg.src = `poses/${classNames[index]}.png`;
      });

      const result = predictions[0].landmarks;
      drawKeypoints(result, predictions[0].annotations);

    }
    rafID = requestAnimationFrame(frameLandmarks);
  };

  frameLandmarks();
};

main();

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;