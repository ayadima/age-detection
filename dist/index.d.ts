import * as tf from '@tensorflow/tfjs-core';
import { BlazeFaceAgeModel } from './face';
export declare function load(blazepath: string, agepath: string, { maxFaces, inputWidth, inputHeight, iouThreshold, scoreThreshold }?: {
    maxFaces?: number;
    inputWidth?: number;
    inputHeight?: number;
    iouThreshold?: number;
    scoreThreshold?: number;
}): Promise<BlazeFaceAgeModel>;
export declare function loadNative(blazehandler: tf.io.IOHandler, ageweights: number, { maxFaces, inputWidth, inputHeight, iouThreshold, scoreThreshold }?: {
    maxFaces?: number;
    inputWidth?: number;
    inputHeight?: number;
    iouThreshold?: number;
    scoreThreshold?: number;
}): Promise<BlazeFaceAgeModel>;
export { NormalizedAgeFace, BlazeFaceAgeModel, BlazeFacePrediction } from './face';
