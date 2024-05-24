---
runme:
  id: 01HWQ66NXVGWC139MAS57E996S
  version: v3
---

# Handpose demo

## Contents

We'll use this demo to collect handpose keypoints for further model tranining.

## Setup

cd into the demo folder:

```sh {"id":"01HWQ66NXVGWC139MARDXSMDQ1"}
cd handpose-keypoints/demo
```

Install dependencies and prepare the build directory:

```sh {"id":"01HWQ66NXVGWC139MARHACGM4P"}
yarn
```

To watch files for changes, and launch a dev server:

```sh {"id":"01HWQ66NXVGWC139MARME49EH6"}
yarn watch
```

## If you are developing handpose locally, and want to test the changes in the demo

Cd into the handpose folder:

```sh {"id":"01HWQ66NXVGWC139MARPA1ZDY6"}
cd handpose-keypoints
```

Install dependencies:

```sh {"id":"01HWQ66NXVGWC139MARSVWWA0V"}
yarn
```

Publish handpose locally:

```sh {"id":"01HWQ66NXVGWC139MARVBEG243"}
yarn build && yarn yalc publish
```

Cd into the demo and install dependencies:

```sh {"id":"01HWQ66NXVGWC139MARW37GJHT"}
cd demo
yarn
```

Link the local handpose to the demo:

```sh {"id":"01HWQ66NXVGWC139MARWXBKXXG"}
yarn yalc link @tensorflow-models/handpose
```

Start the dev demo server:

```sh {"id":"01HWQ66NXVGWC139MAS0BY0DPZ"}
yarn watch
```

To get future updates from the handpose source code:

```sh {"id":"01HWQ66NXVGWC139MAS3EWCVVV"}
# cd up into the handpose directory
cd ../
yarn build && yarn yalc push
```
