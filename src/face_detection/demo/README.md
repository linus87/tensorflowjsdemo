# Demos

Try our demos and get inspired with what you can do with face-detection models!

## Table of Contents
1. [Live Camera Demo](#live-camera-demo)

2. [Upload a Video Demo](#upload-a-video-demo)

3. [How to Run a Demo](#how-to-run-a-demo)

-------------------------------------------------------------------------------

## Live Camera Demo
This demo uses your camera to get live stream and tracks your face in real-time.
You can try out different runtimes to see the difference. It
works on laptops, iPhones and android phones.

[MediaPipeFaceDetector model entry](https://storage.googleapis.com/tfjs-models/demos/face-detection/index.html?model=mediapipe_face_detector)

## Upload a Video Demo
This demo allows you to upload a video (in .mp4 format) to run with the model.
Once the video is processed, it automatically downloads the video with face detection.

[MediaPipeFaceDetector model entry](https://storage.googleapis.com/tfjs-models/demos/face-detection-upload-video/index.html?model=mediapipe_face_detector)

## Before Development
1. Link shared folder.

There is a symbol link "shared", you may need to run below code if you meet compile error hints.
```shell
cd face_defection
ln -s ../../shared src/shared
```
2. Use lcoal static resources

I updated <code>index.html</code> because dat and Stats are from cdnjs. They are not always avaialbe from China.
```text
from:
<script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.6/dat.gui.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/stats.js/r16/Stats.min.js"></script>
to:
<script src="http://localhost:1234/dat.gui.min.js" ></script>
<script src="http://localhost:1234/stats.min.js"></script>
```

### Before Developing Live Video
1. Link shared folder.
There is a symbol link "shared", you may need to run below code if you meet compile error hints.
```shell
cd live_video
ln -s ../../shared src/shared
```

## How to Run a Demo
If you want to run any of the demos locally, follow these steps:

1. Go to the demo folder, e.g. `cd live_video`

2. Remove cache etc. `rm -rf .cache dist node_modules`

3. Build dependency. `yarn build-dep`

4. Install dependencies. `yarn`

5. Run the demo. `yarn watch`

6. The demo runs at `localhost:1234`. (Remember to provide URL model parameter e. g. `localhost:1234/?model=mediapipe_face_detector`)