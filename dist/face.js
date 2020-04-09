"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __read = (this && this.__read) || function (o, n) {
    var m = typeof Symbol === "function" && o[Symbol.iterator];
    if (!m) return o;
    var i = m.call(o), r, ar = [], e;
    try {
        while ((n === void 0 || n-- > 0) && !(r = i.next()).done) ar.push(r.value);
    }
    catch (error) { e = { error: error }; }
    finally {
        try {
            if (r && !r.done && (m = i["return"])) m.call(i);
        }
        finally { if (e) throw e.error; }
    }
    return ar;
};
var __spread = (this && this.__spread) || function () {
    for (var ar = [], i = 0; i < arguments.length; i++) ar = ar.concat(__read(arguments[i]));
    return ar;
};
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var box_1 = require("./box");
var ANCHORS_CONFIG = {
    'strides': [8, 16],
    'anchors': [2, 6]
};
var NUM_LANDMARKS = 6;
function generateAnchors(width, height, outputSpec) {
    var anchors = [];
    for (var i = 0; i < outputSpec.strides.length; i++) {
        var stride = outputSpec.strides[i];
        var gridRows = Math.floor((height + stride - 1) / stride);
        var gridCols = Math.floor((width + stride - 1) / stride);
        var anchorsNum = outputSpec.anchors[i];
        for (var gridY = 0; gridY < gridRows; gridY++) {
            var anchorY = stride * (gridY + 0.5);
            for (var gridX = 0; gridX < gridCols; gridX++) {
                var anchorX = stride * (gridX + 0.5);
                for (var n = 0; n < anchorsNum; n++) {
                    anchors.push([anchorX, anchorY]);
                }
            }
        }
    }
    return anchors;
}
function decodeBounds(boxOutputs, anchors, inputSize) {
    var boxStarts = tf.slice(boxOutputs, [0, 1], [-1, 2]);
    var centers = tf.add(boxStarts, anchors);
    var boxSizes = tf.slice(boxOutputs, [0, 3], [-1, 2]);
    var boxSizesNormalized = tf.div(boxSizes, inputSize);
    var centersNormalized = tf.div(centers, inputSize);
    var halfBoxSize = tf.div(boxSizesNormalized, 2);
    var starts = tf.sub(centersNormalized, halfBoxSize);
    var ends = tf.add(centersNormalized, halfBoxSize);
    var startNormalized = tf.mul(starts, inputSize);
    var endNormalized = tf.mul(ends, inputSize);
    var concatAxis = 1;
    return tf.concat2d([startNormalized, endNormalized], concatAxis);
}
function getInputTensorDimensions(input) {
    return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
        [input.height, input.width];
}
function flipFaceHorizontal(face, imageWidth) {
    var flippedTopLeft, flippedBottomRight, flippedLandmarks;
    if (face.topLeft instanceof tf.Tensor &&
        face.bottomRight instanceof tf.Tensor) {
        var _a = __read(tf.tidy(function () {
            return [
                tf.concat([
                    tf.sub(imageWidth - 1, face.topLeft.slice(0, 1)),
                    face.topLeft.slice(1, 1)
                ]),
                tf.concat([
                    tf.sub(imageWidth - 1, face.bottomRight.slice(0, 1)),
                    face.bottomRight.slice(1, 1)
                ])
            ];
        }), 2), topLeft = _a[0], bottomRight = _a[1];
        flippedTopLeft = topLeft;
        flippedBottomRight = bottomRight;
        if (face.landmarks != null) {
            flippedLandmarks = tf.tidy(function () {
                var a = tf.sub(tf.tensor1d([imageWidth - 1, 0]), face.landmarks);
                var b = tf.tensor1d([1, -1]);
                var product = tf.mul(a, b);
                return product;
            });
        }
    }
    else {
        var _b = __read(face.topLeft, 2), topLeftX = _b[0], topLeftY = _b[1];
        var _c = __read(face.bottomRight, 2), bottomRightX = _c[0], bottomRightY = _c[1];
        flippedTopLeft = [imageWidth - 1 - topLeftX, topLeftY];
        flippedBottomRight = [imageWidth - 1 - bottomRightX, bottomRightY];
        if (face.landmarks != null) {
            flippedLandmarks =
                face.landmarks.map(function (coord) { return ([
                    imageWidth - 1 - coord[0],
                    coord[1]
                ]); });
        }
    }
    var flippedFace = {
        topLeft: flippedTopLeft,
        bottomRight: flippedBottomRight
    };
    if (flippedLandmarks != null) {
        flippedFace.landmarks = flippedLandmarks;
    }
    if (face.probability != null) {
        flippedFace.probability = face.probability instanceof tf.Tensor ?
            face.probability.clone() :
            face.probability;
    }
    return flippedFace;
}
function scaleBoxFromPrediction(face, scaleFactor) {
    return tf.tidy(function () {
        var box;
        if (face.hasOwnProperty('box')) {
            box = face.box;
        }
        else {
            box = face;
        }
        return box_1.scaleBox(box, scaleFactor).startEndTensor.squeeze();
    });
}
function range(num, start, step) {
    return Array(num).fill(0).map(function (_, i) { return start + (i * step); });
}
function getAge(tensor, params, fcageparams) {
    var netinput = tf.tidy(function () {
        var batchTensor = toBatchTensor(tensor, 112, true);
        var meanRgb = [122.782, 117.001, 104.298];
        var normalized = normalize(batchTensor, meanRgb).div(tf.scalar(255));
        var out = tf.relu(conv(normalized, params.entry_flow.conv_in, [2, 2]));
        out = reductionBlock(out, params.entry_flow.reduction_block_0, false);
        out = reductionBlock(out, params.entry_flow.reduction_block_1);
        range(2, 0, 1).forEach(function (idx) {
            out = mainBlock(out, params.middle_flow["main_block_" + idx]);
        });
        out = reductionBlock(out, params.exit_flow.reduction_block);
        out = tf.relu(depthwiseSeparableConv(out, params.exit_flow.separable_conv, [1, 1]));
        return out;
    });
    var pooled = tf.avgPool(netinput, [7, 7], [2, 2], 'valid').as2D(netinput.shape[0], -1);
    var age = fullyConnectedLayer(pooled, fcageparams).as1D();
    return age.dataSync()[0];
}
function padToSquare(imgTensor, isCenterImage) {
    if (isCenterImage === void 0) { isCenterImage = false; }
    return tf.tidy(function () {
        var _a = __read(imgTensor.shape.slice(1), 2), height = _a[0], width = _a[1];
        if (height === width) {
            return imgTensor;
        }
        var dimDiff = Math.abs(height - width);
        var paddingAmount = Math.round(dimDiff * (isCenterImage ? 0.5 : 1));
        var paddingAxis = height > width ? 2 : 1;
        var createPaddingTensor = function (paddingAmount) {
            var paddingTensorShape = imgTensor.shape.slice();
            paddingTensorShape[paddingAxis] = paddingAmount;
            return tf.fill(paddingTensorShape, 0);
        };
        var paddingTensorAppend = createPaddingTensor(paddingAmount);
        var remainingPaddingAmount = dimDiff - (paddingTensorAppend.shape[paddingAxis]);
        var paddingTensorPrepend = isCenterImage && remainingPaddingAmount
            ? createPaddingTensor(remainingPaddingAmount)
            : null;
        var tensorsToStack = [
            paddingTensorPrepend,
            imgTensor,
            paddingTensorAppend
        ]
            .filter(function (t) { return !!t; })
            .map(function (t) { return t.toFloat(); });
        return tf.concat(tensorsToStack, paddingAxis);
    });
}
function toBatchTensor(imgTensor, inputSize, isCenterInputs) {
    if (isCenterInputs === void 0) { isCenterInputs = true; }
    return tf.tidy(function () {
        imgTensor = padToSquare(imgTensor, isCenterInputs);
        if (imgTensor.shape[1] !== inputSize || imgTensor.shape[2] !== inputSize) {
            imgTensor = tf.image.resizeBilinear(imgTensor, [inputSize, inputSize]);
        }
        var inputTensors = imgTensor.as3D(inputSize, inputSize, 3);
        var batchTensor = inputTensors.toFloat().as4D(1, inputSize, inputSize, 3);
        return batchTensor;
    });
}
function fullyConnectedLayer(x, params) {
    return tf.tidy(function () {
        return tf.add(tf.matMul(x, params.weights), params.bias);
    });
}
function depthwiseSeparableConv(x, params, stride) {
    return tf.tidy(function () {
        var out = tf.separableConv2d(x, params.depthwise_filter, params.pointwise_filter, stride, 'same');
        out = tf.add(out, params.bias);
        return out;
    });
}
function normalize(x, meanRgb) {
    return tf.tidy(function () {
        var _a = __read(meanRgb, 3), r = _a[0], g = _a[1], b = _a[2];
        var avg_r = tf.fill(__spread(x.shape.slice(0, 3), [1]), r);
        var avg_g = tf.fill(__spread(x.shape.slice(0, 3), [1]), g);
        var avg_b = tf.fill(__spread(x.shape.slice(0, 3), [1]), b);
        var avg_rgb = tf.concat([avg_r, avg_g, avg_b], 3);
        return tf.sub(x, avg_rgb);
    });
}
function conv(x, params, stride) {
    return tf.add(tf.conv2d(x, params.filters, stride, 'same'), params.bias);
}
function reductionBlock(x, params, isActivateInput) {
    if (isActivateInput === void 0) { isActivateInput = true; }
    var out = isActivateInput ? tf.relu(x) : x;
    out = depthwiseSeparableConv(out, params.separable_conv0, [1, 1]);
    out = depthwiseSeparableConv(tf.relu(out), params.separable_conv1, [1, 1]);
    out = tf.maxPool(out, [3, 3], [2, 2], 'same');
    out = tf.add(out, conv(x, params.expansion_conv, [2, 2]));
    return out;
}
function mainBlock(x, params) {
    var out = depthwiseSeparableConv(tf.relu(x), params.separable_conv0, [1, 1]);
    out = depthwiseSeparableConv(tf.relu(out), params.separable_conv1, [1, 1]);
    out = depthwiseSeparableConv(tf.relu(out), params.separable_conv2, [1, 1]);
    out = tf.add(out, x);
    return out;
}
var BlazeFaceAgeModel = (function () {
    function BlazeFaceAgeModel(model, ageparams, width, height, maxFaces, iouThreshold, scoreThreshold) {
        this.blazeFaceModel = model;
        this.ageModelParams = ageparams;
        this.width = width;
        this.height = height;
        this.maxFaces = maxFaces;
        this.anchorsData = generateAnchors(width, height, ANCHORS_CONFIG);
        this.anchors = tf.tensor2d(this.anchorsData);
        this.inputSizeData = [width, height];
        this.inputSize = tf.tensor1d([width, height]);
        this.iouThreshold = iouThreshold;
        this.scoreThreshold = scoreThreshold;
    }
    BlazeFaceAgeModel.prototype.getAgeModelParams = function () {
        return this.ageModelParams;
    };
    BlazeFaceAgeModel.prototype.getBoundingBoxes = function (inputImage, returnTensors, annotateBoxes) {
        if (annotateBoxes === void 0) { annotateBoxes = true; }
        return __awaiter(this, void 0, void 0, function () {
            var _a, detectedOutputs, boxes, scores, savedConsoleWarnFn, boxIndicesTensor, boxIndices, boundingBoxes, originalHeight, originalWidth, scaleFactor, annotatedBoxes, _loop_1, i;
            var _this = this;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _a = __read(tf.tidy(function () {
                            var resizedImage = inputImage.resizeBilinear([_this.width, _this.height]);
                            var normalizedImage = tf.mul(tf.sub(resizedImage.div(255), 0.5), 2);
                            var batchedPrediction = _this.blazeFaceModel.predict(normalizedImage);
                            var prediction = batchedPrediction.squeeze();
                            var decodedBounds = decodeBounds(prediction, _this.anchors, _this.inputSize);
                            var logits = tf.slice(prediction, [0, 0], [-1, 1]);
                            var scores = tf.sigmoid(logits).squeeze();
                            return [prediction, decodedBounds, scores];
                        }), 3), detectedOutputs = _a[0], boxes = _a[1], scores = _a[2];
                        savedConsoleWarnFn = console.warn;
                        console.warn = function () { };
                        boxIndicesTensor = tf.image.nonMaxSuppression(boxes, scores, this.maxFaces, this.iouThreshold, this.scoreThreshold);
                        console.warn = savedConsoleWarnFn;
                        return [4, boxIndicesTensor.array()];
                    case 1:
                        boxIndices = _b.sent();
                        boxIndicesTensor.dispose();
                        boundingBoxes = boxIndices.map(function (boxIndex) { return tf.slice(boxes, [boxIndex, 0], [1, -1]); });
                        if (!!returnTensors) return [3, 3];
                        return [4, Promise.all(boundingBoxes.map(function (boundingBox) { return __awaiter(_this, void 0, void 0, function () {
                                var vals;
                                return __generator(this, function (_a) {
                                    switch (_a.label) {
                                        case 0: return [4, boundingBox.array()];
                                        case 1:
                                            vals = _a.sent();
                                            boundingBox.dispose();
                                            return [2, vals];
                                    }
                                });
                            }); }))];
                    case 2:
                        boundingBoxes = _b.sent();
                        _b.label = 3;
                    case 3:
                        originalHeight = inputImage.shape[1];
                        originalWidth = inputImage.shape[2];
                        if (returnTensors) {
                            scaleFactor = tf.div([originalWidth, originalHeight], this.inputSize);
                        }
                        else {
                            scaleFactor = [
                                originalWidth / this.inputSizeData[0],
                                originalHeight / this.inputSizeData[1]
                            ];
                        }
                        annotatedBoxes = [];
                        _loop_1 = function (i) {
                            var boundingBox = boundingBoxes[i];
                            var annotatedBox = tf.tidy(function () {
                                var box = boundingBox instanceof tf.Tensor ?
                                    box_1.createBox(boundingBox) :
                                    box_1.createBox(tf.tensor2d(boundingBox));
                                if (!annotateBoxes) {
                                    return box;
                                }
                                var boxIndex = boxIndices[i];
                                var anchor;
                                if (returnTensors) {
                                    anchor = _this.anchors.slice([boxIndex, 0], [1, 2]);
                                }
                                else {
                                    anchor = _this.anchorsData[boxIndex];
                                }
                                var landmarks = tf.slice(detectedOutputs, [boxIndex, NUM_LANDMARKS - 1], [1, -1])
                                    .squeeze()
                                    .reshape([NUM_LANDMARKS, -1]);
                                var probability = tf.slice(scores, [boxIndex], [1]);
                                return { box: box, landmarks: landmarks, probability: probability, anchor: anchor };
                            });
                            annotatedBoxes.push(annotatedBox);
                        };
                        for (i = 0; i < boundingBoxes.length; i++) {
                            _loop_1(i);
                        }
                        boxes.dispose();
                        scores.dispose();
                        detectedOutputs.dispose();
                        return [2, {
                                boxes: annotatedBoxes,
                                scaleFactor: scaleFactor
                            }];
                }
            });
        });
    };
    BlazeFaceAgeModel.prototype.estimateFaces = function (input, returnTensors, flipHorizontal, annotateBoxes) {
        if (returnTensors === void 0) { returnTensors = false; }
        if (flipHorizontal === void 0) { flipHorizontal = false; }
        if (annotateBoxes === void 0) { annotateBoxes = true; }
        return __awaiter(this, void 0, void 0, function () {
            var originalinput, offcanvas, offcontext, _a, width, image, _b, boxes, scaleFactor;
            var _this = this;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        originalinput = input;
                        offcanvas = new OffscreenCanvas(1, 1);
                        offcontext = offcanvas.getContext('2d');
                        _a = __read(getInputTensorDimensions(input), 2), width = _a[1];
                        image = tf.tidy(function () {
                            if (!(input instanceof tf.Tensor)) {
                                input = tf.browser.fromPixels(input);
                            }
                            return input.toFloat().expandDims(0);
                        });
                        return [4, this.getBoundingBoxes(image, returnTensors, annotateBoxes)];
                    case 1:
                        _b = _c.sent(), boxes = _b.boxes, scaleFactor = _b.scaleFactor;
                        image.dispose();
                        if (returnTensors) {
                            return [2, boxes.map(function (face) {
                                    var scaledBox = scaleBoxFromPrediction(face, scaleFactor);
                                    var NormalizedAgeFace = {
                                        topLeft: scaledBox.slice([0], [2]),
                                        bottomRight: scaledBox.slice([2], [2]),
                                        age: 0
                                    };
                                    if (annotateBoxes) {
                                        var _a = face, landmarks = _a.landmarks, probability = _a.probability, anchor = _a.anchor;
                                        var normalizedLandmarks = landmarks.add(anchor).mul(scaleFactor);
                                        NormalizedAgeFace.landmarks = normalizedLandmarks;
                                        NormalizedAgeFace.probability = probability;
                                    }
                                    if (flipHorizontal) {
                                        NormalizedAgeFace = flipFaceHorizontal(NormalizedAgeFace, width);
                                    }
                                    return NormalizedAgeFace;
                                })];
                        }
                        return [2, Promise.all(boxes.map(function (face) { return __awaiter(_this, void 0, void 0, function () {
                                var scaledBox, NormalizedAgeFace, boxData, _a, landmarkData, boxData, probabilityData, anchor_1, _b, scaleFactorX_1, scaleFactorY_1, scaledLandmarks, age, topLeftdata, bottomRightdata, start, end, size, data2, tensor, age;
                                var _this = this;
                                return __generator(this, function (_c) {
                                    switch (_c.label) {
                                        case 0:
                                            scaledBox = scaleBoxFromPrediction(face, scaleFactor);
                                            if (!!annotateBoxes) return [3, 2];
                                            return [4, scaledBox.array()];
                                        case 1:
                                            boxData = _c.sent();
                                            NormalizedAgeFace = {
                                                topLeft: boxData.slice(0, 2),
                                                bottomRight: boxData.slice(2),
                                                age: 0
                                            };
                                            return [3, 4];
                                        case 2: return [4, Promise.all([face.landmarks, scaledBox, face.probability].map(function (d) { return __awaiter(_this, void 0, void 0, function () { return __generator(this, function (_a) {
                                                return [2, d.array()];
                                            }); }); }))];
                                        case 3:
                                            _a = __read.apply(void 0, [_c.sent(), 3]), landmarkData = _a[0], boxData = _a[1], probabilityData = _a[2];
                                            anchor_1 = face.anchor;
                                            _b = __read(scaleFactor, 2), scaleFactorX_1 = _b[0], scaleFactorY_1 = _b[1];
                                            scaledLandmarks = landmarkData
                                                .map(function (landmark) { return ([
                                                (landmark[0] + anchor_1[0]) * scaleFactorX_1,
                                                (landmark[1] + anchor_1[1]) * scaleFactorY_1
                                            ]); });
                                            age = 0;
                                            topLeftdata = boxData.slice(0, 2);
                                            bottomRightdata = boxData.slice(2);
                                            NormalizedAgeFace = {
                                                topLeft: topLeftdata,
                                                bottomRight: bottomRightdata,
                                                landmarks: scaledLandmarks,
                                                probability: probabilityData,
                                                age: age
                                            };
                                            box_1.disposeBox(face.box);
                                            face.landmarks.dispose();
                                            face.probability.dispose();
                                            _c.label = 4;
                                        case 4:
                                            scaledBox.dispose();
                                            if (flipHorizontal) {
                                                NormalizedAgeFace = flipFaceHorizontal(NormalizedAgeFace, width);
                                            }
                                            tf.engine().startScope();
                                            if (originalinput instanceof ImageData) {
                                                offcanvas.width = originalinput.width;
                                                offcanvas.height = originalinput.height;
                                                offcontext.putImageData(originalinput, 0, 0);
                                                start = NormalizedAgeFace.topLeft;
                                                end = NormalizedAgeFace.bottomRight;
                                                size = [end[0] - start[0], end[1] - start[1]];
                                                offcontext.drawImage(offcanvas, start[0], start[1], size[0], size[1], 0, 0, size[0], size[1]);
                                                data2 = offcontext.getImageData(0, 0, size[0], size[1]);
                                                tensor = tf.browser.fromPixels(data2).expandDims(0).toFloat();
                                                age = getAge(tensor, this.ageModelParams.FaceFeatureParams.params, this.ageModelParams.classifierParams.params.fc.age);
                                                NormalizedAgeFace.age = age;
                                                tf.dispose(age);
                                                tensor.dispose();
                                                offcontext.clearRect(0, 0, offcanvas.width, offcanvas.height);
                                            }
                                            tf.engine().endScope();
                                            return [2, NormalizedAgeFace];
                                    }
                                });
                            }); }))];
                }
            });
        });
    };
    return BlazeFaceAgeModel;
}());
exports.BlazeFaceAgeModel = BlazeFaceAgeModel;
//# sourceMappingURL=face.js.map