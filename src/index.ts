import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import {BlazeFaceAgeModel} from './face';
import * as native from '@tensorflow/tfjs-react-native'

async function fetchNetWeights(uri: string): Promise<Float32Array> {
  return new Float32Array(await (await fetchOrThrow(uri)).arrayBuffer())
}

async function fetchOrThrow(
  uri: string,
  init?: RequestInit
): Promise<Response> {

  const res = await fetch(uri, init)
  if (!(res.status < 400)) {
    throw new Error(`failed to fetch: (${res.status}) ${res.statusText}, from url: ${res.url}`)
  }
  return res
}

function extractParams(weights: Float32Array) {

  const classifierWeightSize = (512 * 1 + 1) + (512 * 2 + 2)

  const featureExtractorWeights = weights.slice(0, weights.length - classifierWeightSize)
  const classifierWeights = weights.slice(weights.length - classifierWeightSize)

  return {
    classifierParams : extractClassifierParams(classifierWeights),
    FaceFeatureParams : extractFaceFeatureParams(featureExtractorWeights)
  }
}

function extractClassifierParams(weights: Float32Array): { params: Object} {

  const {
    extractWeights,
    getRemainingWeights
  } = extractWeightsFactory(weights)

  const extractFCParams = extractFCParamsFactory(extractWeights)

  const age = extractFCParams(512, 1)
  const gender = extractFCParams(512, 2)

  if (getRemainingWeights().length !== 0) {
    throw new Error(`weights remaing after extract: ${getRemainingWeights().length}`)
  }

  return {
    params: { fc: { age, gender } }
  }
}

function range(num : number, start :  number, step : number){
  return Array(num).fill(0).map((_, i) => start + (i * step))
}

function extractFaceFeatureParams(weights: Float32Array, numMainBlocks: number = 2): { params: Object} {

  const {
    extractWeights,
    getRemainingWeights
  } = extractWeightsFactory(weights)

  const {
    extractConvParams,
    extractSeparableConvParams,
    extractReductionBlockParams,
    extractMainBlockParams
  } = extractorsFactory(extractWeights)

  const entry_flow_conv_in = extractConvParams(3, 32, 3)
  const entry_flow_reduction_block_0 = extractReductionBlockParams(32, 64)
  const entry_flow_reduction_block_1 = extractReductionBlockParams(64, 128)

  const entry_flow = {
    conv_in: entry_flow_conv_in,
    reduction_block_0: entry_flow_reduction_block_0,
    reduction_block_1: entry_flow_reduction_block_1
  }

  const middle_flow = {}
  range(numMainBlocks, 0, 1).forEach((idx) => {
    middle_flow[`main_block_${idx}`] = extractMainBlockParams(128)
  })

  const exit_flow_reduction_block = extractReductionBlockParams(128, 256)
  const exit_flow_separable_conv = extractSeparableConvParams(256, 512)

  const exit_flow = {
    reduction_block: exit_flow_reduction_block,
    separable_conv: exit_flow_separable_conv
  }

  if (getRemainingWeights().length !== 0) {
    throw new Error(`weights remaing after extract: ${getRemainingWeights().length}`)
  }

  return {
    params: { entry_flow, middle_flow, exit_flow }
  }
}

function extractConvParamsFactory(
  extractWeights: Function
) {

  return function(
    channelsIn: number,
    channelsOut: number,
    filterSize: number
  ): Object {

    const filters = tf.tensor4d(
      extractWeights(channelsIn * channelsOut * filterSize * filterSize),
      [filterSize, filterSize, channelsIn, channelsOut]
    )
    const bias = tf.tensor1d(extractWeights(channelsOut))

    return { filters, bias }
  }

}

function extractorsFactory(extractWeights: Function) {

  const extractConvParams = extractConvParamsFactory(extractWeights)
  const extractSeparableConvParams = extractSeparableConvParamsFactory(extractWeights)

  function extractReductionBlockParams(channelsIn: number, channelsOut: number): Object {

    const separable_conv0 = extractSeparableConvParams(channelsIn, channelsOut)
    const separable_conv1 = extractSeparableConvParams(channelsOut, channelsOut)
    const expansion_conv = extractConvParams(channelsIn, channelsOut, 1)

    return { separable_conv0, separable_conv1, expansion_conv }
  }

  function extractMainBlockParams(channels: number): Object {

    const separable_conv0 = extractSeparableConvParams(channels, channels)
    const separable_conv1 = extractSeparableConvParams(channels, channels)
    const separable_conv2 = extractSeparableConvParams(channels, channels)

    return { separable_conv0, separable_conv1, separable_conv2 }
  }

  return {
    extractConvParams,
    extractSeparableConvParams,
    extractReductionBlockParams,
    extractMainBlockParams
  }

}

function extractSeparableConvParamsFactory(
  extractWeights: Function
) {

  return function(channelsIn: number, channelsOut: number) {
    const depthwise_filter = tf.tensor4d(extractWeights(3 * 3 * channelsIn), [3, 3, channelsIn, 1])
    const pointwise_filter = tf.tensor4d(extractWeights(channelsIn * channelsOut), [1, 1, channelsIn, channelsOut])
    const bias = tf.tensor1d(extractWeights(channelsOut))


    return {
      depthwise_filter,
      pointwise_filter,
      bias
    }
  }

}

function extractWeightsFactory(weights: Float32Array) {
  let remainingWeights = weights

  function extractWeights(numWeights: number): Float32Array {
    const ret = remainingWeights.slice(0, numWeights)
    remainingWeights = remainingWeights.slice(numWeights)
    return ret
  }

  function getRemainingWeights(): Float32Array {
    return remainingWeights
  }

  return {
    extractWeights,
    getRemainingWeights
  }
}

function extractFCParamsFactory(
  extractWeights: Function
) {

  return function(
    channelsIn: number,
    channelsOut: number
  ): Object {

    const fc_weights = tf.tensor2d(extractWeights(channelsIn * channelsOut), [channelsIn, channelsOut])
    const fc_bias = tf.tensor1d(extractWeights(channelsOut))


    return {
      weights: fc_weights,
      bias: fc_bias
    }
  }

}

/**
 * Load blazeface.
 *
 * @param config A configuration object with the following properties:
 *  `maxFaces` The maximum number of faces returned by the model.
 *  `inputWidth` The width of the input image.
 *  `inputHeight` The height of the input image.
 *  `iouThreshold` The threshold for deciding whether boxes overlap too
 * much.
 *  `scoreThreshold` The threshold for deciding when to remove boxes based
 * on score.
 */
export async function load(blazepath:string, agepath:string, blazemodelJson? : tf.io.ModelJSON, blazemodelWeights? : number, agemodelWeights? : number,{
  maxFaces = 10,
  inputWidth = 128,
  inputHeight = 128,
  iouThreshold = 0.3,
  scoreThreshold = 0.75
} = {}): Promise<BlazeFaceAgeModel> {
  if (tfconv == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  let blazeface;
  let weights;
  let ageparams;

  if(blazemodelJson && blazemodelWeights && agemodelWeights){
    blazeface = await tfconv.loadGraphModel(native.bundleResourceIO(blazemodelJson, blazemodelWeights));
    ageparams = extractParams(new Float32Array(agemodelWeights))
  } else {
    blazeface = await tfconv.loadGraphModel(blazepath);
    weights = await fetchNetWeights(agepath)
    ageparams = extractParams(weights)
  }


  const model = new BlazeFaceAgeModel(
      blazeface, ageparams, inputWidth, inputHeight, maxFaces, iouThreshold,
      scoreThreshold);
  return model;
}

export {NormalizedAgeFace, BlazeFaceAgeModel, BlazeFacePrediction} from './face';