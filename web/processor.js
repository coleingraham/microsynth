/**
 * microsynth AudioWorklet processor.
 *
 * Loads the raw WASM module (no wasm-bindgen) and calls the
 * #[no_mangle] C exports directly:
 *   ms_init(sample_rate)
 *   ms_alloc(size) -> ptr
 *   ms_free(ptr, size)
 *   ms_compile(source_ptr, source_len) -> 0|1
 *   ms_render(out_left_ptr, out_right_ptr)
 */
class MicrosynthProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();

        this.ready = false;
        this.wasm = null;
        this.leftPtr = 0;
        this.rightPtr = 0;

        // The main thread sends us WASM bytes and DSL source via the port.
        this.port.onmessage = (e) => this.handleMessage(e.data);

        // If WASM bytes were passed via processorOptions, init immediately.
        if (options.processorOptions && options.processorOptions.wasmBytes) {
            this.initWasm(options.processorOptions.wasmBytes);
        }
    }

    async handleMessage(msg) {
        switch (msg.type) {
            case 'init':
                await this.initWasm(msg.wasmBytes);
                break;
            case 'compile':
                this.compileDSL(msg.source);
                break;
            case 'stop':
                this.ready = false;
                break;
        }
    }

    async initWasm(wasmBytes) {
        try {
            // Instantiate raw WASM module (no wasm-bindgen, no imports needed)
            const module = await WebAssembly.compile(wasmBytes);
            const instance = await WebAssembly.instantiate(module, {});
            this.wasm = instance.exports;

            // Initialize engine at the AudioContext sample rate
            this.wasm.ms_init(sampleRate);

            // Allocate output buffers in WASM memory (128 f32s = 512 bytes each)
            this.leftPtr = this.wasm.ms_alloc(128 * 4);
            this.rightPtr = this.wasm.ms_alloc(128 * 4);

            this.port.postMessage({ type: 'ready' });
        } catch (err) {
            this.port.postMessage({ type: 'error', message: 'WASM init failed: ' + err.message });
        }
    }

    compileDSL(source) {
        if (!this.wasm) {
            this.port.postMessage({ type: 'error', message: 'WASM not initialized' });
            return;
        }

        // Write the source string into WASM memory
        const encoder = typeof TextEncoder !== 'undefined'
            ? new TextEncoder()
            : { encode: (s) => new Uint8Array([...s].map(c => c.charCodeAt(0))) };
        const encoded = encoder.encode(source);
        const srcPtr = this.wasm.ms_alloc(encoded.length);

        // Copy into WASM memory
        const memory = new Uint8Array(this.wasm.memory.buffer);
        memory.set(encoded, srcPtr);

        // Compile
        const result = this.wasm.ms_compile(srcPtr, encoded.length);

        // Free the source buffer
        this.wasm.ms_free(srcPtr, encoded.length);

        if (result === 0) {
            this.ready = true;
            this.port.postMessage({ type: 'compiled' });
        } else {
            this.ready = false;
            this.port.postMessage({ type: 'error', message: 'DSL compilation failed' });
        }
    }

    process(inputs, outputs, parameters) {
        if (!this.ready || !this.wasm) {
            // Output silence
            return true;
        }

        const output = outputs[0];
        if (!output || output.length === 0) return true;

        // Render 128 samples
        this.wasm.ms_render(this.leftPtr, this.rightPtr);

        // Create Float32Array views over the WASM output buffers.
        // Must re-create views each time in case memory grew.
        const leftBuf = new Float32Array(this.wasm.memory.buffer, this.leftPtr, 128);
        const rightBuf = new Float32Array(this.wasm.memory.buffer, this.rightPtr, 128);

        // Copy to output channels
        if (output.length >= 1) output[0].set(leftBuf);
        if (output.length >= 2) output[1].set(rightBuf);

        return true;
    }
}

registerProcessor('microsynth-processor', MicrosynthProcessor);
