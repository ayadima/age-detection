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
Object.defineProperty(exports, "__esModule", { value: true });
var tfconv = require("@tensorflow/tfjs-converter");
var tf = require("@tensorflow/tfjs-core");
var face_1 = require("./face");
function fetchNetWeights(uri) {
    return __awaiter(this, void 0, void 0, function () {
        var _a;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    _a = Float32Array.bind;
                    return [4, fetchOrThrow(uri)];
                case 1: return [4, (_b.sent()).arrayBuffer()];
                case 2: return [2, new (_a.apply(Float32Array, [void 0, _b.sent()]))()];
            }
        });
    });
}
function fetchOrThrow(uri, init) {
    return __awaiter(this, void 0, void 0, function () {
        var res;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4, fetch(uri, init)];
                case 1:
                    res = _a.sent();
                    if (!(res.status < 400)) {
                        throw new Error("failed to fetch: (" + res.status + ") " + res.statusText + ", from url: " + res.url);
                    }
                    return [2, res];
            }
        });
    });
}
function extractParams(weights) {
    var classifierWeightSize = (512 * 1 + 1) + (512 * 2 + 2);
    var featureExtractorWeights = weights.slice(0, weights.length - classifierWeightSize);
    var classifierWeights = weights.slice(weights.length - classifierWeightSize);
    return {
        classifierParams: extractClassifierParams(classifierWeights),
        FaceFeatureParams: extractFaceFeatureParams(featureExtractorWeights)
    };
}
function extractClassifierParams(weights) {
    var _a = extractWeightsFactory(weights), extractWeights = _a.extractWeights, getRemainingWeights = _a.getRemainingWeights;
    var extractFCParams = extractFCParamsFactory(extractWeights);
    var age = extractFCParams(512, 1);
    var gender = extractFCParams(512, 2);
    if (getRemainingWeights().length !== 0) {
        throw new Error("weights remaing after extract: " + getRemainingWeights().length);
    }
    return {
        params: { fc: { age: age, gender: gender } }
    };
}
function range(num, start, step) {
    return Array(num).fill(0).map(function (_, i) { return start + (i * step); });
}
function extractFaceFeatureParams(weights, numMainBlocks) {
    if (numMainBlocks === void 0) { numMainBlocks = 2; }
    var _a = extractWeightsFactory(weights), extractWeights = _a.extractWeights, getRemainingWeights = _a.getRemainingWeights;
    var _b = extractorsFactory(extractWeights), extractConvParams = _b.extractConvParams, extractSeparableConvParams = _b.extractSeparableConvParams, extractReductionBlockParams = _b.extractReductionBlockParams, extractMainBlockParams = _b.extractMainBlockParams;
    var entry_flow_conv_in = extractConvParams(3, 32, 3);
    var entry_flow_reduction_block_0 = extractReductionBlockParams(32, 64);
    var entry_flow_reduction_block_1 = extractReductionBlockParams(64, 128);
    var entry_flow = {
        conv_in: entry_flow_conv_in,
        reduction_block_0: entry_flow_reduction_block_0,
        reduction_block_1: entry_flow_reduction_block_1
    };
    var middle_flow = {};
    range(numMainBlocks, 0, 1).forEach(function (idx) {
        middle_flow["main_block_" + idx] = extractMainBlockParams(128);
    });
    var exit_flow_reduction_block = extractReductionBlockParams(128, 256);
    var exit_flow_separable_conv = extractSeparableConvParams(256, 512);
    var exit_flow = {
        reduction_block: exit_flow_reduction_block,
        separable_conv: exit_flow_separable_conv
    };
    if (getRemainingWeights().length !== 0) {
        throw new Error("weights remaing after extract: " + getRemainingWeights().length);
    }
    return {
        params: { entry_flow: entry_flow, middle_flow: middle_flow, exit_flow: exit_flow }
    };
}
function extractConvParamsFactory(extractWeights) {
    return function (channelsIn, channelsOut, filterSize) {
        var filters = tf.tensor4d(extractWeights(channelsIn * channelsOut * filterSize * filterSize), [filterSize, filterSize, channelsIn, channelsOut]);
        var bias = tf.tensor1d(extractWeights(channelsOut));
        return { filters: filters, bias: bias };
    };
}
function extractorsFactory(extractWeights) {
    var extractConvParams = extractConvParamsFactory(extractWeights);
    var extractSeparableConvParams = extractSeparableConvParamsFactory(extractWeights);
    function extractReductionBlockParams(channelsIn, channelsOut) {
        var separable_conv0 = extractSeparableConvParams(channelsIn, channelsOut);
        var separable_conv1 = extractSeparableConvParams(channelsOut, channelsOut);
        var expansion_conv = extractConvParams(channelsIn, channelsOut, 1);
        return { separable_conv0: separable_conv0, separable_conv1: separable_conv1, expansion_conv: expansion_conv };
    }
    function extractMainBlockParams(channels) {
        var separable_conv0 = extractSeparableConvParams(channels, channels);
        var separable_conv1 = extractSeparableConvParams(channels, channels);
        var separable_conv2 = extractSeparableConvParams(channels, channels);
        return { separable_conv0: separable_conv0, separable_conv1: separable_conv1, separable_conv2: separable_conv2 };
    }
    return {
        extractConvParams: extractConvParams,
        extractSeparableConvParams: extractSeparableConvParams,
        extractReductionBlockParams: extractReductionBlockParams,
        extractMainBlockParams: extractMainBlockParams
    };
}
function extractSeparableConvParamsFactory(extractWeights) {
    return function (channelsIn, channelsOut) {
        var depthwise_filter = tf.tensor4d(extractWeights(3 * 3 * channelsIn), [3, 3, channelsIn, 1]);
        var pointwise_filter = tf.tensor4d(extractWeights(channelsIn * channelsOut), [1, 1, channelsIn, channelsOut]);
        var bias = tf.tensor1d(extractWeights(channelsOut));
        return {
            depthwise_filter: depthwise_filter,
            pointwise_filter: pointwise_filter,
            bias: bias
        };
    };
}
function extractWeightsFactory(weights) {
    var remainingWeights = weights;
    function extractWeights(numWeights) {
        var ret = remainingWeights.slice(0, numWeights);
        remainingWeights = remainingWeights.slice(numWeights);
        return ret;
    }
    function getRemainingWeights() {
        return remainingWeights;
    }
    return {
        extractWeights: extractWeights,
        getRemainingWeights: getRemainingWeights
    };
}
function extractFCParamsFactory(extractWeights) {
    return function (channelsIn, channelsOut) {
        var fc_weights = tf.tensor2d(extractWeights(channelsIn * channelsOut), [channelsIn, channelsOut]);
        var fc_bias = tf.tensor1d(extractWeights(channelsOut));
        return {
            weights: fc_weights,
            bias: fc_bias
        };
    };
}
function load(blazepath, agepath, _a) {
    var _b = _a === void 0 ? {} : _a, _c = _b.maxFaces, maxFaces = _c === void 0 ? 10 : _c, _d = _b.inputWidth, inputWidth = _d === void 0 ? 128 : _d, _e = _b.inputHeight, inputHeight = _e === void 0 ? 128 : _e, _f = _b.iouThreshold, iouThreshold = _f === void 0 ? 0.3 : _f, _g = _b.scoreThreshold, scoreThreshold = _g === void 0 ? 0.75 : _g;
    return __awaiter(this, void 0, void 0, function () {
        var blazeface, weights, ageparams, model;
        return __generator(this, function (_h) {
            switch (_h.label) {
                case 0:
                    if (tfconv == null) {
                        throw new Error("Cannot find TensorFlow.js. If you are using a <script> tag, please " +
                            "also include @tensorflow/tfjs on the page before using this model.");
                    }
                    return [4, tfconv.loadGraphModel(blazepath)];
                case 1:
                    blazeface = _h.sent();
                    return [4, fetchNetWeights(agepath)];
                case 2:
                    weights = _h.sent();
                    ageparams = extractParams(weights);
                    model = new face_1.BlazeFaceAgeModel(blazeface, ageparams, inputWidth, inputHeight, maxFaces, iouThreshold, scoreThreshold);
                    return [2, model];
            }
        });
    });
}
exports.load = load;
function loadNative(blazehandler, ageweights, _a) {
    var _b = _a === void 0 ? {} : _a, _c = _b.maxFaces, maxFaces = _c === void 0 ? 10 : _c, _d = _b.inputWidth, inputWidth = _d === void 0 ? 128 : _d, _e = _b.inputHeight, inputHeight = _e === void 0 ? 128 : _e, _f = _b.iouThreshold, iouThreshold = _f === void 0 ? 0.3 : _f, _g = _b.scoreThreshold, scoreThreshold = _g === void 0 ? 0.75 : _g;
    return __awaiter(this, void 0, void 0, function () {
        var blazeface, ageparams, model;
        return __generator(this, function (_h) {
            switch (_h.label) {
                case 0:
                    if (tfconv == null) {
                        throw new Error("Cannot find TensorFlow.js. If you are using a <script> tag, please " +
                            "also include @tensorflow/tfjs on the page before using this model.");
                    }
                    return [4, tfconv.loadGraphModel(blazehandler)];
                case 1:
                    blazeface = _h.sent();
                    ageparams = extractParams(new Float32Array(ageweights));
                    model = new face_1.BlazeFaceAgeModel(blazeface, ageparams, inputWidth, inputHeight, maxFaces, iouThreshold, scoreThreshold);
                    return [2, model];
            }
        });
    });
}
exports.loadNative = loadNative;
var face_2 = require("./face");
exports.BlazeFaceAgeModel = face_2.BlazeFaceAgeModel;
//# sourceMappingURL=index.js.map