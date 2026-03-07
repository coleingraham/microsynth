/**
 * microsynth AudioWorklet processor.
 *
 * Loads the raw WASM module (no wasm-bindgen) and calls the
 * #[no_mangle] C exports directly:
 *   ms_init(sample_rate)
 *   ms_alloc(size) -> ptr
 *   ms_free(ptr, size)
 *   ms_compile(source_ptr, source_len) -> 0|1
 *   ms_compile_def(source_ptr, source_len) -> 0|1
 *   ms_spawn_voice() -> voice_id (u64, 0=error)
 *   ms_voice_gate(voice_id, value)
 *   ms_voice_param(voice_id, param_ptr, param_len, value)
 *   ms_free_voice(voice_id)
 *   ms_free_done() -> count
 *   ms_render(out_left_ptr, out_right_ptr)
 */
class MicrosynthProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();

        this.ready = false;
        this.wasm = null;
        this.leftPtr = 0;
        this.rightPtr = 0;
        this.frameCount = 0;

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
            case 'compileDef':
                this.compileDef(msg.source);
                break;
            case 'spawnVoice':
                this.spawnVoice(msg.id);
                break;
            case 'voiceGate':
                this.voiceGate(msg.voiceId, msg.value);
                break;
            case 'voiceParam':
                this.voiceParam(msg.voiceId, msg.param, msg.value);
                break;
            case 'freeVoice':
                this.freeVoice(msg.voiceId);
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

    writeString(str) {
        const encoder = typeof TextEncoder !== 'undefined'
            ? new TextEncoder()
            : { encode: (s) => new Uint8Array([...s].map(c => c.charCodeAt(0))) };
        const encoded = encoder.encode(str);
        const ptr = this.wasm.ms_alloc(encoded.length);
        const memory = new Uint8Array(this.wasm.memory.buffer);
        memory.set(encoded, ptr);
        return { ptr, len: encoded.length };
    }

    compileDSL(source) {
        if (!this.wasm) {
            this.port.postMessage({ type: 'error', message: 'WASM not initialized' });
            return;
        }

        const { ptr: srcPtr, len } = this.writeString(source);
        const result = this.wasm.ms_compile(srcPtr, len);
        this.wasm.ms_free(srcPtr, len);

        if (result === 0) {
            this.ready = true;
            this.port.postMessage({ type: 'compiled' });
        } else {
            this.ready = false;
            this.port.postMessage({ type: 'error', message: 'DSL compilation failed' });
        }
    }

    compileDef(source) {
        if (!this.wasm) {
            this.port.postMessage({ type: 'error', message: 'WASM not initialized' });
            return;
        }

        const { ptr: srcPtr, len } = this.writeString(source);
        const result = this.wasm.ms_compile_def(srcPtr, len);
        this.wasm.ms_free(srcPtr, len);

        if (result === 0) {
            this.ready = true;
            this.port.postMessage({ type: 'defCompiled' });
        } else {
            this.ready = false;
            this.port.postMessage({ type: 'error', message: 'DSL compilation failed' });
        }
    }

    spawnVoice(requestId) {
        if (!this.wasm) return;
        const voiceId = Number(this.wasm.ms_spawn_voice());
        this.port.postMessage({ type: 'voiceSpawned', id: requestId, voiceId });
    }

    voiceGate(voiceId, value) {
        if (!this.wasm) return;
        this.wasm.ms_voice_gate(voiceId, value);
    }

    voiceParam(voiceId, param, value) {
        if (!this.wasm) return;
        const { ptr, len } = this.writeString(param);
        this.wasm.ms_voice_param(voiceId, ptr, len, value);
        this.wasm.ms_free(ptr, len);
    }

    freeVoice(voiceId) {
        if (!this.wasm) return;
        this.wasm.ms_free_voice(voiceId);
    }

    process(inputs, outputs, parameters) {
        if (!this.ready || !this.wasm) {
            // Output silence
            return true;
        }

        const output = outputs[0];
        if (!output || output.length === 0) return true;

        // Periodically free finished voices
        this.frameCount++;
        if (this.frameCount % 16 === 0) {
            this.wasm.ms_free_done();
        }

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
