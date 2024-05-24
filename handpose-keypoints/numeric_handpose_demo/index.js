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

const classNames = ['fist', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
console.log(tf);
const numericHanposeModel = tf.sequential();
numericHanposeModel.add(tf.layers.inputLayer({inputShape: [5, 3], dtype: 'float32', batchSize: 1}));
numericHanposeModel.add(tf.layers.activation({activation: 'relu'}));
numericHanposeModel.add(tf.layers.flatten());
const weights = tf.tensor([[2.6339848 , -0.7821255, -0.9852441, 0.0480192  , -0.3466824 , -1.9451324, -0.2665296, 0.4633622 , 0.5863307 , 1.4673749 ],
  [1.9798787 , -3.6038599, -4.460216 , 0.2952887  , -10.3301592, 2.3445635 , -0.143085 , 0.92823   , 2.5617113 , 2.1121659 ],
  [1.9838854 , -0.3016337, -0.5679148, -4.4008055 , -2.9695716 , 0.0452375 , 0.2540358 , 0.3761422 , 0.0191029 , 0.3871876 ],
  [-5.0452185, 0.655054  , 0.7981847 , 0.4678645  , 0.6775135  , -0.3385722, -0.3506645, -1.8684748, -1.1486729, 3.8903823 ],
  [-8.430047 , 1.0430573 , 0.1428999 , -12.6395006, 0.5788171  , -0.1291275, -8.076951 , 1.1151588 , 0.273112  , 0.7584921 ],
  [-0.3173599, 0.5262296 , -0.0265811, -0.4357024 , -0.1919112 , -2.4167848, 0.0672482 , 0.8349618 , -0.6821571, 1.6574062 ],
  [-0.0781015, 0.9455061 , 0.3056812 , 0.8599624  , 0.1987738  , -0.3718994, -4.5338659, -0.1285613, -0.7152275, -2.9902966],
  [-3.3314159, -4.2465887, 4.936142  , 3.0950744  , 0.9306732  , -0.0423123, -3.0542874, 3.0736477 , -2.4638669, -1.128652 ],
  [0.9750291 , 0.3274792 , -0.0958808, 0.118009   , 0.4248115  , -1.2034907, 1.4910746 , 1.3296365 , -0.7900974, -2.8479481],
  [-1.4355721, 0.406817  , 0.7897013 , 1.3567253  , 0.705112   , 0.0477447 , -3.2462018, -0.7236025, 0.5025509 , -5.1215072],
  [1.6338153 , -2.3408177, -6.9712372, 3.397702   , 2.5956841  , 3.6143422 , -1.8373487, -0.3265107, -1.8837547, 1.1698129 ],
  [1.8685191 , 0.5671845 , 0.5649589 , 0.3055698  , 0.1246265  , 0.7307346 , 1.4903445 , -2.0175452, 0.1810325 , -1.7679509],
  [-4.8163428, 0.1706621 , 0.3639418 , 0.5388684  , 1.2080667  , 0.6291083 , 3.2750857 , -3.0049524, 0.7873013 , -3.3101127],
  [-0.9758605, -1.1150088, 1.296903  , 1.4453139  , 1.4802957  , 2.7682326 , 6.4073501 , -2.96071  , -4.3682613, -0.7844887],
  [1.6052431 , 0.0245548 , -1.1945288, -0.017442  , 0.2870344  , 0.3435654 , 1.4639882 , -1.0085355, -0.3355551, -1.1747116]]
);
const bias = tf.tensor([2.6990962, 0.1053372, -0.4259273, -0.4504739, 0.0568584, -2.4289062, -0.7209694, 0.1070402, -0.5185959, 1.0869477]
);
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

      const handposePredicts = numericHanposeModel.predict(convertVectorsIntoAngles(predictions[0].annotations).reshape([1, 5, 3]));

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