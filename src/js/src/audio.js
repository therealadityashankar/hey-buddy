/** @module audio */

/**
 * A class that batches audio samples and calls a callback with the batch.
 */
export class AudioBatcher {
    /**
     * @param {number} batchSeconds - The number of seconds to batch.
     * @param {number} batchIntervalSeconds - The number of seconds to wait before calling the callback.
     * @param {string} workletUrl - The URL of the worklet to use.
     * @param {string} workletName - The name of the worklet to use.
     * @param {number} workletTargetSampleRate - The target sample rate of the worklet.
     */
    constructor(
        batchSeconds=2.0,
        batchIntervalSeconds=0.05, // 50ms
        workletUrl="/worklet.js",
        workletName="hey-buddy",
        workletTargetSampleRate=16000,
    ) {
        this.initialized = false;
        this.callbacks = [];
        this.batchSeconds = batchSeconds;
        this.batchIntervalSeconds = batchIntervalSeconds;
        this.batchIntervalCount = 0;
        this.workletUrl = workletUrl;
        this.workletName = workletName;
        this.workletTargetSampleRate = workletTargetSampleRate;
        this.buffer = new Float32Array(this.batchSamples);
        this.buffer.fill(0);
        this.initialize();
    }

    /**
     * The number of samples in a batch.
     * @type {number}
     */
    get batchSamples() {
        return Math.floor(this.batchSeconds * this.workletTargetSampleRate);
    }

    /**
     * The number of samples in a batch interval.
     * @type {number}
     */
    get batchIntervalSamples() {
        return Math.floor(this.batchIntervalSeconds * this.workletTargetSampleRate);
    }

    /**
     * Pushes new audio samples into the buffer.
     * @param {Float32Array} data - The new audio samples.
     */
    push(data) {
        const dataLength = data.length;
        // Shift the buffer back by this much
        this.buffer.set(this.buffer.subarray(dataLength));
        // Append the new data
        this.buffer.set(data, this.buffer.length - dataLength);
        this.batchIntervalCount += dataLength;
        // If we have enough samples, call the callbacks and reset the interval count
        if (this.batchIntervalCount >= this.batchIntervalSamples) {
            this.callbacks.forEach(callback => callback(this.buffer));
            this.batchIntervalCount = 0;
        }
    }

    /**
     * Adds a callback to be called with each batch.
     * @param {Function} callback - The callback to add.
     */
    onBatch(callback) {
        this.callbacks.push(callback);
    }

    /**
     * Removes a callback from the list of callbacks.
     * @param {Function} callback - The callback to remove.
     */
    offBatch(callback) {
        this.callbacks = this.callbacks.filter(c => c !== callback);
    }

    /**
     * Initializes the audio batcher.
     */
    async initialize() {
        if (this.initialized) {
            return;
        }
        this.stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                autoGainControl: true,
                noiseSuppression: true,
            }
        });
        this.audioContext = new AudioContext();
        this.sourceNode = new MediaStreamAudioSourceNode(
            this.audioContext,
            { mediaStream: this.stream }
        );
        this.workerNode = await AudioNode.create(
            this.audioContext,
            this.workletUrl,
            this.workletName,
            this.workletTargetSampleRate,
        );
        this.sourceNode.connect(this.workerNode.worker);
        this.workerNode.worker.port.onmessage = (event) => {
            this.push(event.data);
        }
        this.initialized = true;
    }
}

/**
 * A class that wraps an AudioWorkletNode.
 */
export class AudioNode {
    /**
     * @param {AudioContext} context - The audio context.
     * @param {AudioWorkletNode} worker - The audio worklet node.
     */
    constructor(context, worker) {
        this.context = context;
        this.worker = worker;
    }

    /**
     * Creates an AudioNode.
     * @param {AudioContext} context - The audio context.
     * @param {string} workletUrl - The URL of the worklet to use.
     * @param {string} workletName - The name of the worklet to use.
     * @param {number} workletTargetSampleRate - The target sample rate of the worklet.
     * @returns {Promise<AudioNode>} The created AudioNode.
     */
    static async create(context, workletUrl, workletName, workletTargetSampleRate) {
        await context.audioWorklet.addModule(workletUrl);
        const workletOptions = {
            processorOptions: {
                targetSampleRate: workletTargetSampleRate,
            }
        };
        const worker = new AudioWorkletNode(context, workletName, workletOptions);
        return new AudioNode(context, worker);
    }
}
