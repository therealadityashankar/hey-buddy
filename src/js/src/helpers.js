/**
 * Check if an object is 'empty'.
 *
 * @param object $o The object to check.o
 * @return bool True if the object is empty.
 */
export let isEmpty = (o) => {
    return (
        o === null ||
        o === undefined ||
        o === '' ||
        o === 'null' ||
        (Array.isArray(o) && o.length === 0) ||
        (typeof o === 'object' &&
            o.constructor.name === 'Object' &&
            Object.getOwnPropertyNames(o).length === 0)
    );
};

/**
 * Generate a random UUID.
 *
 * @return string A random UUID.
 */
export let uuidv4 = () => {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
        (+c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> +c / 4).toString(16)
    );
}

/**
 * Merge multiple typed arrays into a single typed array.
 * Assumes that all the arrays are of the same type.
 *
 * @param {Array<TypedArray>} arrays - The arrays to merge. Any kind of typed array is allowed.
 * @returns {TypedArray} - The merged typed array.
 */
export let mergeTypedArrays = (arrays) => {
    let totalLength = arrays.reduce((acc, array) => acc + array.length, 0);
    let result = new arrays[0].constructor(totalLength);
    let offset = 0;
    arrays.forEach((array) => {
        result.set(array, offset);
        offset += array.length;
    });
    return result;
}

/**
 * Binds a method to window mousemove, and then unbinds it when released
 * or when the mouse leaves the window.
 */
export let bindPointerUntilRelease = (callback, releaseCallback = null) => {
    let onWindowPointerMove = (e) => {
        callback(e);
    }
    let onWindowPointerUpOrLeave = (e) => {
        if (!isEmpty(releaseCallback)) {
            releaseCallback(e);
        }
        window.removeEventListener("mouseup", onWindowPointerUpOrLeave, true);
        window.removeEventListener("mouseleave", onWindowPointerUpOrLeave, true);
        window.removeEventListener("touchend", onWindowPointerUpOrLeave, true);
        window.removeEventListener("mousemove", onWindowPointerMove, true);
        window.removeEventListener("touchmove", onWindowPointerMove, true);
    }
    window.addEventListener("mouseup", onWindowPointerUpOrLeave, true);
    window.addEventListener("mouseleave", onWindowPointerUpOrLeave, true);
    window.addEventListener("touchend", onWindowPointerUpOrLeave, true);
    window.addEventListener("mousemove", onWindowPointerMove, true);
    window.addEventListener("touchmove", onWindowPointerMove, true);
};

/**
 * Binds drag events to an element.
 * The callback is called with an object containing the following properties:
 * - start: The starting point of the drag.
 *   - x: The x coordinate.
 *   - y: The y coordinate.
 * - current: The current point of the drag.
 *   - x: The x coordinate.
 *   - y: The y coordinate.
 * - delta: The difference between the current and starting points.
 *   - x: The x coordinate.
 *   - y: The y coordinate.
 * - startEvent: The event that started the drag.
 * - moveEvent: The event that triggered the callback.
 * The releaseCallback is called when the drag is released.
 */
export let bindPointerDrag = (element, startCallback, callback, releaseCallback = null) => {
    const pointerStart = (e) => {
        if (e.type === "mousedown" && e.button !== 0) {
            return;
        }
        e.preventDefault();
        const startPosition = e.type === "mousedown" ? e : e.touches[0];
        const startPoint = {x: startPosition.clientX, y: startPosition.clientY};
        if (!isEmpty(startCallback)) {
            startCallback({
                start: startPoint,
                startEvent: e
            });
        }
        bindPointerUntilRelease(
            (e2) => {
                const currentPosition = e2.type === "mousemove" ? e2 : e2.touches[0];
                const currentPoint = {x: currentPosition.clientX, y: currentPosition.clientY};
                const delta = {x: currentPoint.x - startPoint.x, y: currentPoint.y - startPoint.y};
                callback({
                    start: startPoint,
                    current: currentPoint,
                    delta: delta,
                    startEvent: e,
                    moveEvent: e2
                });
            },
            (e2) => {
                if (!isEmpty(releaseCallback)) {
                    releaseCallback({
                        start: startPoint,
                        startEvent: e,
                        releaseEvent: e2
                    });
                }
            }
        );
    };
    element.addEventListener("mousedown", pointerStart);
    element.addEventListener("touchstart", pointerStart);
};

/**
 * Returns a promise that resolves after a given number of milliseconds.
 */
export let sleep = (ms) => {
    return new Promise(resolve => setTimeout(resolve, ms));
};

/**
 * Play audio samples using the Web Audio API.
 * @param {Float32Array} audioSamples - The audio samples to play.
 * @param {number} sampleRate - The sample rate of the audio samples.
 */
export let playAudioSamples = (audioSamples, sampleRate = 16000) => {
    // Create an AudioContext
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();

    // Create an AudioBuffer
    const audioBuffer = audioContext.createBuffer(
        1, // number of channels
        audioSamples.length, // length of the buffer in samples
        sampleRate // sample rate (samples per second)
    );

    // Fill the AudioBuffer with the Float32Array of audio samples
    audioBuffer.getChannelData(0).set(audioSamples);

    // Create a BufferSource node
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;

    // Connect the source to the AudioContext's destination (the speakers)
    source.connect(audioContext.destination);

    // Start playback
    source.start();
};
