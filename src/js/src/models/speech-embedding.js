/** @module models/speech-embedding */
import { ONNX } from "../onnx.js";
import { ONNXModel } from "./base.js";
import { MelSpectrogram } from "./mel-spectrogram.js";

/**
 * Speech Embedding model
 * @extends ONNXModel
 */
export class SpeechEmbedding extends ONNXModel {
    /**
     * Constructor
     * @param {string} modelPath - Path to the ONNX model
     * @param {MelSpectrogram} spectrogramModel - Mel spectrogram model
     * @param {number} spectrogramMelBins - Number of Mel bins for the Mel spectrogram model
     * @param {number} embeddingDim - Dimension of the embeddings
     * @param {number} windowSize - Size of the window
     * @param {number} windowStride - Stride of the window
     * @param {number} sampleRate - Sample rate of the input audio
     */
    constructor(
        modelPath,
        power = 0,
        webnn = 1,
        webgpu = 2,
        webgl = 3,
        wasm = 4,
        embeddingDim = 96,
        windowSize = 76,
        windowStride = 8,
    ) {
        super(
            modelPath,
            power,
            webnn,
            webgpu,
            webgl,
            wasm
        );
        this.embeddingDim = embeddingDim;
        this.windowSize = windowSize;
        this.windowStride = windowStride;
    }

    /** 
     * Test the model
     * @param {boolean} debug - Debug mode
     * @throws {Error} - If the model fails the test
     */
    async test(debug = false) {
        const melTensor = await ONNX.createTensor(
            "float32",
            new Float32Array(new Array(100 * 32).fill(0)),
            [100, 32]
        );
        let result = await this.run(melTensor);
        if (result.dims.length === 2 && 
            result.dims[0] === 4 &&
            result.dims[1] === 96
        ) {
            if (debug) {
                console.log(`Speech embedding model OK, executed in ${this.duration} ms`);
            }
        } else {
            console.error("Unexpected speech embedding result", result);
            throw new Error("Speech embedding model failed");
        }
    }

    /**
     * Execute the model
     * @param {Float32Array} input - Input data
     * @returns {Promise} - Promise that resolves with the output of the model, which is a 2D array
     * @throws {Error} - If the input data is not a Float32Array
     */
    async execute(spectrograms) {
        const [numFrames, melBins] = spectrograms.dims;
        if (numFrames < this.windowSize) {
            throw new Error(`Audio is too short to process - require ${this.windowSize} samples, got ${numFrames}`);
        }
        // Calculate the number of batches
        const numTruncatedFrames = numFrames - (numFrames - this.windowSize) % this.windowStride;
        const numBatches = (numTruncatedFrames - this.windowSize) / this.windowStride + 1;

        // Create buffer for output
        const embeddings = await ONNX.createTensor(
            "float32",
            (new Array(numBatches * this.embeddingDim)).fill(0),
            [numBatches, this.embeddingDim]
        );

        // Iterate through windows
        const windowBatches = [];
        for (
            let windowStart = 0;
            windowStart < numTruncatedFrames - this.windowSize + this.windowStride;
            windowStart += this.windowStride
        ) {
            const windowEnd = windowStart + this.windowSize;
            const windowTensor = await ONNX.createTensor(
                "float32",
                spectrograms.data.slice(windowStart * melBins, windowEnd * melBins),
                [this.windowSize, melBins, 1]
            );
            windowBatches.push([windowStart, windowEnd, windowTensor]);
        }

        // Restack windows into a single tensor
        const stackedWindowTensor = await ONNX.createTensor(
            "float32",
            new Float32Array(numBatches * this.windowSize * melBins),
            [numBatches, this.windowSize, melBins, 1]
        );
        for (let i = 0; i < numBatches; i++) {
            stackedWindowTensor.data.set(windowBatches[i][2].data, i * this.windowSize * melBins);
        }

        // Execute the model
        // TODO: Determine why this takes so much longer in the browser than it does in python
        const output = await this.session.run({ input_1: stackedWindowTensor });

        for (let i = 0; i < numBatches; i++) {
            embeddings.data.set(
                output.conv2d_19.data.slice(
                    i * this.embeddingDim,
                    (i + 1) * this.embeddingDim
                ),
                i * this.embeddingDim
            );
        }
        return embeddings;
    }
}
