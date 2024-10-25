/** @module models/vad */
import { ONNX } from "../onnx.js";
import { ONNXModel } from "./base.js";

/**
 * Silero VAD model
 * @extends ONNXModel
 */
export class SileroVAD extends ONNXModel {
    /**
     * Constructor
     * @param {string} modelPath - Path to the ONNX model
     * @param {number} sampleRate - Sample rate of the input audio
     */
    constructor(
        modelPath = "/pretrained/silero-vad.onnx",
        power = 0,
        webnn = 1,
        webgpu = 2,
        webgl = 3,
        wasm = 4,
        sampleRate = 16000,
    ) {
        super(
            modelPath,
            power,
            webnn,
            webgpu,
            webgl,
            wasm,
        );
        this.sampleRate = sampleRate || 16000;
    }

    /**
     * Test the model
     * @param {boolean} debug - If true, log the result to the console
     * @throws {Error} - If the model fails the test
     */
    async test(debug = false) {
        let result = await this.run(new Float32Array(16000).fill(0));
        if (!isNaN(result) && 0.0 <= result && result <= 1.0) {
            if (debug) {
                console.log(`VAD model OK, executed in ${this.duration} ms`);
            }
        } else {
            throw new Error(`VAD model failed - got ${result}`);
        }
    }

    /**
     * Execute the model
     * @param {Float32Array} input - Input data
     * @returns {Promise} - Promise that resolves with the output of the model, which is a single float
     * @throws {Error} - If the input data is not a Float32Array
     */
    async execute(input) {
        if (this.h === undefined || this.c === undefined || this.sr === undefined) {
            this.sr = await ONNX.createTensor("int64", [this.sampleRate], [1]);
            this.h = await ONNX.createTensor("float32", (new Array(128)).fill(0), [2, 1, 64]);
            this.c = await ONNX.createTensor("float32", (new Array(128)).fill(0), [2, 1, 64]);
        }
        const inputTensor = await ONNX.createTensor("float32", input, [1, input.length]);
        const output = await this.session.run({
            input: inputTensor,
            h: this.h,
            c: this.c,
            sr: this.sr,
        });
        this.c = output.cn;
        this.h = output.hn;
        return output.output.data[0];
    }
}
