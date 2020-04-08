# Face and Age Detection

Face and Age detection model that aims to localize, identify and distinguish different faces with age prediction in a single image.

This project uses [**Blazeface**](https://github.com/tensorflow/tfjs-models/tree/master/blazeface) model from Tensorflow.js models to detect faces and [**face-api.js**]'(https://github.com/justadudewhohacks/face-api.js) age classifier to predict age.
 
This TensorFlow.js model does not require you to know about machine learning.
It can take input as any browser-based image elements (`<img>`, `<video>`, `<canvas>`
elements, for example) and returns an array of bounding boxes with class name and confidence level.

## Usage

There are one main way to get this model in your JavaScript project : by installing it from NPM and using a build tool like Parcel, WebPack, or Rollup.

### via NPM (or yarn)

```sh
npm install agedetection
```
or 
```sh
yarn add agedetection
```

```js
// Note: you do not need to import @tensorflow/tfjs here.

import * as faceage from 'agedetection';

const img = document.getElementById('img');

// Load the model.
const model = await faceage.load(PATH_TO_JSON_BLAZEFACE_MODEL, PATH_TO_WEIGHTS_FACEAPI_AGE_MODEL);

// Classify the image.
const predictions = await model.estimatefaces(img);

console.log('Predictions: ');
console.log(predictions);
```

## API

#### Loading the model
`agedetection` is the module name. When using ES6 imports, `faceage` is the module.

```ts

faceage.load(PATH_TO_JSON_BLAZEFACE_MODEL, PATH_TO_WEIGHTS_FACEAPI_AGE_MODEL);
```

Args:
**PATH_TO_JSON_BLAZEFACE_MODEL** string that specifies json file containing blazeface model as input of the model. This file can be an url or a locally stored file.
**PATH_TO_WEIGHTS_FACEAPI_AGE_MODEL** string that specifies weights file containing face-api weights of age detection model as input of the model. This file can be an url or a locally stored file.

Returns a `model` object.

#### Detecting Faces with Age prediction

You can detect faces with age predictions with the model without needing to create a Tensor.
`model.estimatefaces` takes an input image element and returns an array of bounding boxes around the face with the predicted age.

This method exists on the model that is loaded from `faceage.load`.

```ts
model.estimatefaces(
  img: tf.Tensor3D | ImageData | HTMLImageElement |
      HTMLCanvasElement | HTMLVideoElement
)
```

Args:

**img:** A Tensor or an image element to make a detection on.

Returns an array of classes and probabilities that looks like:

```js
[{
  topLeft: 145,
  bottomRight: 300,
  age : 17
}, {
  topLeft: 300,
  bottomRight: 450,
  age : 25
}]
```
