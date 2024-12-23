/** @module hey-buddy */
import { ONNX } from "./onnx.js";
import { AudioBatcher } from "./audio.js";
import {
    SileroVAD,
    SpeechEmbedding,
    MelSpectrogram,
    WakeWord
} from "./models.js";

/**
 * HeyBuddy class for running wake word detection.
 */
export class HeyBuddy {
    /**
     * Create a HeyBuddy instance.
     * @param {Object} [options] - Options object.
     * @param {number} [options.positiveVadThreshold=0.5] - VAD threshold for speech.
     * @param {number} [options.negativeVadThreshold=0.25] - VAD threshold for silence.
     * @param {number} [options.negativeVadCount=8] - Number of negative VADs to trigger silence.
     * @param {number} [options.wakeWordThreads=4] - Number of threads for wake word detection.
     * @param {number} [options.wakeWordThreshold=0.5] - Wake word detection threshold.
     * @param {string|string[]} [options.modelPath="/models/hey-buddy.onnx"] - Path to wake word model.
     * @param {string} [options.vadModelPath="/pretrained/silero-vad.onnx"] - Path to VAD model.
     * @param {string} [options.embeddingModelPath="/pretrained/speech-embedding.onnx"] - Path to speech embedding model.
     * @param {string} [options.spectrogramModelPath="/pretrained/mel-spectrogram.onnx"] - Path to mel spectrogram model.
     * @param {number} [options.batchSeconds=1.08] - Number of seconds per batch.
     * @param {number} [options.batchIntervalSeconds=0.12] - Number of seconds between batches.
     * @param {number} [options.targetSampleRate=16000] - Target sample rate for audio.
     * @param {number} [options.spectrogramMelBins=32] - Number of mel bins for spectrogram.
     * @param {number} [options.embeddingDim=96] - Dimension of speech embedding.
     * @param {number} [options.embeddingWindowSize=76] - Window size for speech embedding.
     * @param {number} [options.embeddingWindowStride=8] - Window stride for speech embedding.
     */
    constructor (options) {
        options = options || {};
        // Get options or use defaults for runtime
        this.debug = options.debug || false;
        this.positiveVadThreshold = options.positiveVadThreshold || 0.65;
        this.negativeVadThreshold = options.negativeVadThreshold || 0.4;
        this.negativeVadCount = options.negativeVadCount || 8;
        this.wakeWordThreads = options.wakeWordThreads || 4;
        this.wakeWordThreshold = options.wakeWordThreshold || 0.5;
        this.wakeWordInterval = options.wakeWordInterval || 2.0; // How often a wake word can be uttered

        // Get options or use defaults for models
        const modelPath = options.modelPath || "/models/hey-buddy.onnx";
        const modelArray = Array.isArray(modelPath) ? modelPath : [modelPath];
        const vadModelPath = options.vadModelPath || "/pretrained/silero-vad.onnx";
        const embeddingModelPath = options.embeddingModelPath || "/pretrained/speech-embedding.onnx";
        const spectrogramModelPath = options.spectrogramModelPath || "/pretrained/mel-spectrogram.onnx";
        const batchSeconds = options.batchSeconds || 1.08; // 1080ms * 16khz = 17280 samples
        const batchIntervalSeconds = options.batchIntervalSeconds || 0.12; // 120ms * 16khz = 1920 samples
        const targetSampleRate = options.targetSampleRate || 16000;
        const spectrogramMelBins = options.spectrogramMelBins || 32;
        const embeddingDim = options.embeddingDim || 96;
        const embeddingWindowSize = options.embeddingWindowSize || 76;
        const embeddingWindowStride = options.embeddingWindowStride || 8;
        const wakeWordEmbeddingFrames = options.wakeWordEmbeddingFrames || 16;

        // Initialize shared models
        this.vad = new SileroVAD(vadModelPath);
        this.vad.test(this.debug);

        this.spectrogram = new MelSpectrogram(spectrogramModelPath);
        this.spectrogram.test(this.debug);
        this.spectrogramMelBins = spectrogramMelBins;
        this.spectrogramBuffer = null;

        this.embedding = new SpeechEmbedding(
            embeddingModelPath,
            embeddingDim,
            embeddingWindowSize,
            embeddingWindowStride,
        );
        this.embedding.test(this.debug);
        this.embeddingDim = embeddingDim;
        this.embeddingWindowSize = embeddingWindowSize;
        this.embeddingWindowStride = embeddingWindowStride;
        this.embeddingBuffer = null;

        // Initialize wake word models
        this.wakeWords = {};
        this.wakeWordTimes = {};
        this.wakeWordEmbeddingFrames = wakeWordEmbeddingFrames;
        for (let model of modelArray) {
            let modelName = model.split("/").pop().split(".")[0];
            this.wakeWords[modelName] = new WakeWord(model);
            this.wakeWords[modelName].test(this.debug);
        }

        // Initialize state
        this.listening = false;
        this.negatives = 0;
        this.recording = false;
        this.audioBuffer = null;
        this.frameIntervalEma = 0;
        this.frameIntervalEmaWeight = 0.1;
        this.frameTimeEma = 0;
        this.frameTimeEmaWeight = 0.1;

        this.speechStartCallbacks = [];
        this.speechEndCallbacks = [];
        this.recordingCallbacks = [];
        this.processedCallbacks = [];
        this.detectedCallbacks = [];

        // Initialize batcher and add callback
        this.batcher = new AudioBatcher(
            batchSeconds,
            batchIntervalSeconds,
            targetSampleRate
        );
        this.batcher.onBatch((batch) => this.process(batch));
    }

    /**
     * Gets the names of wake words, chunked for threaded wake word detection.
     * @returns {string[][]} - Names of wake words.
     */
    get chunkedWakeWords() {
        return Object.keys(this.wakeWords).reduce((carry, name, i) => {
            const chunkIndex = Math.floor(i / this.wakeWordThreads);
            if (!carry[chunkIndex]) {
                carry[chunkIndex] = [];
            }
            carry[chunkIndex].push(name);
            return carry;
        }, []);
    }

    /**
     * Add a callback for when a wake word is detected.
     * @param {string|string[]} names - Name of wake word.
     * @param {Function} callback - Callback function.
     */
    onDetected(names, callback) {
        this.detectedCallbacks.push({names, callback});
    }

    /**
     * Add a callback for processed data.
     * @param {Function} callback - Callback function.
     */
    onProcessed(callback) {
        this.processedCallbacks.push(callback);
    }

    /**
     * Add a callback for speech start.
     * @param {Function} callback - Callback function.
     */
    onSpeechStart(callback) {
        this.speechStartCallbacks.push(callback);
    }

    /**
     * Add a callback for speech end.
     * @param {Function} callback - Callback function.
     */
    onSpeechEnd(callback) {
        this.speechEndCallbacks.push(callback);
    }

    /**
     * Add a callback for recording.
     * @param {Function} callback - Callback function.
     */
    onRecording(callback) {
        this.recordingCallbacks.push(callback);
    }

    /**
     * Trigger speech start event.
     */
    speechStart() {
        if (this.debug) {
            console.log("Speech start");
        }
        for (let callback of this.speechStartCallbacks) {
            callback();
        }
    }

    /**
     * Trigger speech end event.
     */
    speechEnd() {
        if (this.debug) {
            console.log("Speech end");
        }
        for (let callback of this.speechEndCallbacks) {
            callback();
        }
        if (this.recording) {
            this.dispatchRecording();
            this.recording = false;
        }
    }

    /**
     * Dispatch recording to all recording callbacks.
     */ 
    dispatchRecording() {
        if (this.audioBuffer === null) {
            console.error("No recording to dispatch");
            return;
        }
        if (this.debug) {
            const recordingLength = this.audioBuffer.length;
            const recordedDuration = recordingLength / this.batcher.targetSampleRate;
            console.log(`Dispatching recording with ${recordingLength} frames (${recordedDuration} s)`);
        }
        for (let callback of this.recordingCallbacks) {
            callback(this.audioBuffer);
        }
        this.audioBuffer = null;
    }

    /**
     * Trigger wake word detection event.
     * @param {string} name - Name of wake word.
     */
    wakeWordDetected(name) {
        const now = Date.now();
        if (this.wakeWordTimes[name] && (now - this.wakeWordTimes[name]) < this.wakeWordInterval * 1000) {
            return;
        }
        if (this.debug) {
            console.log("Wake word detected:", name);
        }
        this.recording = true;
        this.wakeWordTimes[name] = now;

        for (let {names, callback} of this.detectedCallbacks) {
            if (Array.isArray(names) && names.includes(name) || names === name) {
                callback();
            }
        }
    }

    /**
     * Trigger processed event.
     * @param {Object} data - Processed data.
     */
    processed(data) {
        for (let callback of this.processedCallbacks) {
            callback(data);
        }
    }

    /**
     * Runs wake word detection on a subset of wake words.
     * @param {string[]} wakeWordNames - Names of wake words to check.
     * @returns {Promise} - Promise that resolves when wake word detection is complete.
     */
    async checkWakeWordSubset(wakeWordNames) {
        return await Promise.all(
            wakeWordNames.map(name => this.wakeWords[name].run(this.embeddingBuffer))
        );
    }

    /**
     * Run wake word detection on audio.
     * @returns {Promise} - Promise that resolves when wake word detection is complete.
     */
    async checkWakeWords() {
        const returnMap = {};
        for (let nameChunk of this.chunkedWakeWords) {
            const wakeWordProbabilities = await this.checkWakeWordSubset(nameChunk);
            for (let i = 0; i < nameChunk.length; i++) {
                const name = nameChunk[i];
                const probability = wakeWordProbabilities[i];
                returnMap[name] = probability;
            }
        }
        for (let name in returnMap) {
            if (returnMap[name] > this.wakeWordThreshold) {
                this.wakeWordDetected(name);
            }
        }
        return returnMap;
    }

    /**
     * Process audio batch.
     * @param {Float32Array} audio - Audio samples.
     */
    async process(audio) {
        // Start timer
        this.frameStart = (new Date()).getTime();

        let timeSinceLastFrame;
        if (this.frameEnd !== undefined && this.frameEnd !== null) {
            this.frameInterval = this.frameStart - this.frameEnd;
        } else {
            this.frameInterval = 0;
        }
        if (this.frameIntervalEma === 0) {
            this.frameIntervalEma = this.frameInterval;
        } else {
            this.frameIntervalEma = this.frameIntervalEma * (1 - this.frameIntervalEmaWeight) + this.frameInterval * this.frameIntervalEmaWeight;
        }

        // Get the last batch of samples
        const lastBatch = audio.subarray(audio.length - this.batcher.batchIntervalSamples);

        // Run VAD on it
        const speechProbability = await this.vad.run(lastBatch);
        const hasSpeech = speechProbability > this.positiveVadThreshold;
        const hasSilence = speechProbability < this.negativeVadThreshold;

        // Calculate the spectrogram for this buffer, assert it is exactly one window
        const spectrograms = await this.spectrogram.run(audio);
        this.spectrogramBuffer = await ONNX.createTensor(
            "float32",
            spectrograms.data,
            spectrograms.dims.slice(2)
        );
        
        // Calculate new embedding, assert it is one embedding frame
        const embedding = await this.embedding.run(this.spectrogramBuffer);

        // Push the embedding into the buffer
        if (this.embeddingBuffer === null) {
            this.embeddingBuffer = await ONNX.createTensor(
                "float32",
                embedding.data,
                [embedding.dims[embedding.dims.length-2], this.embeddingDim]
            );
        } else {
            const toShift = this.embeddingBuffer.dims[0] + embedding.dims[0] - this.wakeWordEmbeddingFrames;
            // Shift back
            if (toShift > 0) {
                if (this.embeddingBuffer.dims[0] < this.wakeWordEmbeddingFrames) {
                    const embeddingData = new Float32Array(this.wakeWordEmbeddingFrames * this.embeddingDim);
                    embeddingData.set(this.embeddingBuffer.data.subarray(toShift * this.embeddingDim));
                    embeddingData.set(embedding.data, this.wakeWordEmbeddingFrames - embedding.dims[0]);
                    this.embeddingBuffer = await ONNX.createTensor(
                        "float32",
                        embeddingData,
                        [this.wakeWordEmbeddingFrames, this.embeddingDim]
                    );
                } else {
                    this.embeddingBuffer.data.set(this.embeddingBuffer.data.subarray(toShift * this.embeddingDim));
                    this.embeddingBuffer.data.set(embedding.data, this.embeddingBuffer.length - this.embeddingDim);
                }
            } else {
                // Append
                const embeddingData = new Float32Array(this.embeddingBuffer.data.length + embedding.data.length);
                embeddingData.set(this.embeddingBuffer.data);
                embeddingData.set(embedding.data, this.embeddingBuffer.data.length);
                this.embeddingBuffer = await ONNX.createTensor(
                    "float32",
                    embeddingData,
                    [this.embeddingBuffer.dims[0] + embedding.dims[0], this.embeddingDim]
                );
            }
        }

        // Debounce VAD negatives and trigger events
        if (!hasSpeech) {
            if (hasSilence) {
                this.negatives += 1;
            }
            if (this.negatives > this.negativeVadCount) {
                if (this.listening) {
                    this.speechEnd();
                }
                this.listening = false;
            }
        } else {
            this.negatives = 0;
            if (!this.listening) {
                this.speechStart();
            }
            this.listening = true;
        }

        if (this.listening && this.embeddingBuffer.dims[0] === this.wakeWordEmbeddingFrames) {
            // If we're listening, run wake word detection
            const probabilities = await this.checkWakeWords();
            // Trigger callbacks with processed data
            this.processed({
                listening: true,
                recording: this.recording,
                speech: {probability: speechProbability, active: hasSpeech},
                wakeWords: Object.entries(probabilities).reduce(
                    (carry, [name, probability]) => {
                        carry[name] = {
                            probability,
                            active: probability > this.wakeWordThreshold
                        };
                        return carry;
                    },
                    {}
                )
            });
        } else {
            // Trigger callbacks right away if we're not listening
            this.processed({
                listening: false,
                recording: this.recording,
                speech: {probability: speechProbability, active: hasSpeech},
                wakeWords: Object.entries(this.wakeWords).reduce(
                    (carry, [name, model]) => {
                        carry[name] = {
                            probability: 0.0,
                            active: false
                        };
                        return carry;
                    },
                    {}
                )
            });
        }

        // If we're recording, append audio to buffer
        if (this.recording) {
            if (this.audioBuffer === null) {
                this.audioBuffer = new Float32Array(audio.length);
                this.audioBuffer.set(audio);
            } else {
                const concatenated = new Float32Array(this.audioBuffer.length + lastBatch.length);
                concatenated.set(this.audioBuffer);
                concatenated.set(lastBatch, this.audioBuffer.length);
                this.audioBuffer = concatenated;
            }
        }

        // Stop timer
        this.frameEnd = (new Date()).getTime();
        this.frameTime = this.frameEnd - this.frameStart;
        if (this.frameTimeEma === 0) {
            this.frameTimeEma = this.frameTime;
        } else {
            this.frameTimeEma = this.frameTimeEma * (1 - this.frameTimeEmaWeight) + this.frameTime * this.frameTimeEmaWeight;
        }
    }
};

if (typeof window !== "undefined") {
    window.HeyBuddy = HeyBuddy;
}
