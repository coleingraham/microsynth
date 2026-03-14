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

const midiToFreq = (midi) => 440 * Math.pow(2, (midi - 69) / 12);
class MicrosynthProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();

        this.ready = false;
        this.wasm = null;
        this.leftPtr = 0;
        this.rightPtr = 0;
        this.frameCount = 0;
        this.seq = null;  // null when not playing

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
            case 'initBus':
                this.initBus();
                break;
            case 'registerDef':
                this.registerDef(msg.name, msg.source);
                break;
            case 'spawnVoiceNamed':
                this.spawnVoiceNamed(msg.id, msg.name);
                break;
            case 'seqStartShell':
                this.seqStartShell(msg);
                break;
            case 'seqStartComposite':
                this.seqStartComposite(msg);
                break;
            case 'seqStop':
                this.seqStopPlayback();
                break;
            case 'seqUpdate':
                this.seqUpdatePlayback(msg);
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
        this.wasm.ms_voice_gate(BigInt(voiceId), value);
    }

    voiceParam(voiceId, param, value) {
        if (!this.wasm) return;
        const { ptr, len } = this.writeString(param);
        this.wasm.ms_voice_param(BigInt(voiceId), ptr, len, value);
        this.wasm.ms_free(ptr, len);
    }

    freeVoice(voiceId) {
        if (!this.wasm) return;
        this.wasm.ms_free_voice(BigInt(voiceId));
    }

    initBus() {
        if (!this.wasm) {
            this.port.postMessage({ type: 'error', message: 'WASM not initialized' });
            return;
        }
        this.wasm.ms_init_with_bus(sampleRate);
        this.ready = true;
        this.port.postMessage({ type: 'busReady' });
    }

    registerDef(name, source) {
        if (!this.wasm) {
            this.port.postMessage({ type: 'error', message: 'WASM not initialized' });
            return;
        }
        const { ptr: namePtr, len: nameLen } = this.writeString(name);
        const { ptr: srcPtr, len: srcLen } = this.writeString(source);
        const result = this.wasm.ms_register_def(namePtr, nameLen, srcPtr, srcLen);
        this.wasm.ms_free(namePtr, nameLen);
        this.wasm.ms_free(srcPtr, srcLen);

        if (result === 0) {
            this.port.postMessage({ type: 'defRegistered', name });
        } else {
            this.port.postMessage({ type: 'error', message: `Failed to register def: ${name}` });
        }
    }

    spawnVoiceNamed(requestId, name) {
        if (!this.wasm) return;
        const { ptr: namePtr, len: nameLen } = this.writeString(name);
        const voiceId = Number(this.wasm.ms_spawn_voice_named(namePtr, nameLen));
        this.wasm.ms_free(namePtr, nameLen);
        this.port.postMessage({ type: 'voiceSpawned', id: requestId, voiceId });
    }

    spawnVoiceNamedSync(name) {
        if (!this.wasm) return 0;
        const { ptr, len } = this.writeString(name);
        const voiceId = Number(this.wasm.ms_spawn_voice_named(ptr, len));
        this.wasm.ms_free(ptr, len);
        return voiceId;
    }

    seqStartShell(msg) {
        // Stop any active sequencer first
        this.seqStopPlayback();

        const { steps, bpm, subdivision, instrumentName, instrumentType } = msg;
        const divisorMap = { '8n': 2, '16n': 4, '32n': 8 };
        const divisor = divisorMap[subdivision] || 4;
        const stepDurationSamples = Math.round(sampleRate * 60 / (bpm * divisor));

        let activeVoice = 0;
        if (instrumentType === 'sustaining') {
            activeVoice = this.spawnVoiceNamedSync(instrumentName);
        }

        this.seq = {
            kind: 'shell',
            steps,
            bpm,
            subdivision,
            instrumentName,
            instrumentType,
            stepDurationSamples,
            sampleCounter: stepDurationSamples, // fire first step immediately
            stepIndex: 0,
            activeVoice,
        };
    }

    seqStartComposite(msg) {
        // Stop any active sequencer first
        this.seqStopPlayback();

        const { cells, bpm, subdivision, tonicMidi, voiceName } = msg;
        const divisorMap = { '8n': 2, '16n': 4, '32n': 8 };
        const divisor = divisorMap[subdivision] || 4;
        const stepDurationSamples = Math.round(sampleRate * 60 / (bpm * divisor));

        // Pre-spawn a sustaining voice
        const activeVoice = this.spawnVoiceNamedSync(voiceName);

        this.seq = {
            kind: 'composite',
            cells,
            bpm,
            subdivision,
            tonicMidi,
            voiceName,
            stepDurationSamples,
            sampleCounter: stepDurationSamples, // fire first step immediately
            stepIndex: 0,
            activeVoice,
        };
    }

    seqStopPlayback() {
        if (!this.seq) return;
        const seq = this.seq;

        if (seq.activeVoice && seq.activeVoice !== 0) {
            if (this.wasm) {
                this.wasm.ms_voice_gate(BigInt(seq.activeVoice), 0);
                this.wasm.ms_free_voice(BigInt(seq.activeVoice));
            }
        }

        this.seq = null;
        this.port.postMessage({ type: 'seqStopped' });
    }

    seqUpdatePlayback(msg) {
        if (!this.seq) return;
        const seq = this.seq;

        // Update step/cell data in place — no position reset
        if (msg.steps !== undefined) seq.steps = msg.steps;
        if (msg.cells !== undefined) seq.cells = msg.cells;
        if (msg.tonicMidi !== undefined) seq.tonicMidi = msg.tonicMidi;
        if (msg.instrumentName !== undefined) seq.instrumentName = msg.instrumentName;

        if (msg.bpm !== undefined || msg.subdivision !== undefined) {
            if (msg.bpm !== undefined) seq.bpm = msg.bpm;
            if (msg.subdivision !== undefined) seq.subdivision = msg.subdivision;
            const divisorMap = { '8n': 2, '16n': 4, '32n': 8 };
            const divisor = divisorMap[seq.subdivision] || 4;
            const newDuration = Math.round(sampleRate * 60 / (seq.bpm * divisor));
            // Clamp sampleCounter so we don't overshoot the new duration
            if (seq.sampleCounter > newDuration) {
                seq.sampleCounter = newDuration;
            }
            seq.stepDurationSamples = newDuration;
        }

        // Only reset position if the new array is shorter than current position
        const totalSteps = seq.kind === 'shell'
            ? (seq.steps ? seq.steps.length : 0)
            : (seq.cells ? seq.cells.length : 0);
        if (totalSteps > 0 && seq.stepIndex >= totalSteps) {
            seq.stepIndex = 0;
        }
    }

    seqApplyShellStep(seq, stepIdx) {
        const step = seq.steps[stepIdx];

        if (step === 1) {
            // Onset
            if (seq.instrumentType === 'percussive') {
                const voiceId = this.spawnVoiceNamedSync(seq.instrumentName);
                if (voiceId !== 0) {
                    this.wasm.ms_voice_gate(BigInt(voiceId), 1);
                }
            } else {
                // Sustaining: gate off old, free, spawn new, gate on
                if (seq.activeVoice && seq.activeVoice !== 0) {
                    this.wasm.ms_voice_gate(BigInt(seq.activeVoice), 0);
                    this.wasm.ms_free_voice(BigInt(seq.activeVoice));
                }
                const voiceId = this.spawnVoiceNamedSync(seq.instrumentName);
                if (voiceId !== 0) {
                    this.wasm.ms_voice_gate(BigInt(voiceId), 1);
                }
                seq.activeVoice = voiceId;
            }
        } else if (step === 0) {
            // Rest
            if (seq.instrumentType === 'sustaining' && seq.activeVoice && seq.activeVoice !== 0) {
                this.wasm.ms_voice_gate(BigInt(seq.activeVoice), 0);
                this.wasm.ms_free_voice(BigInt(seq.activeVoice));
                seq.activeVoice = 0;
            }
        }
        // step === 2 (sustain): no-op
    }

    seqApplyCompositeStep(seq, stepIdx) {
        const cell = seq.cells[stepIdx];
        if (!cell || seq.activeVoice === 0) return;

        const voiceId = BigInt(seq.activeVoice);

        if (cell.onset) {
            const midi = seq.tonicMidi + cell.pitch_index;
            const freq = midiToFreq(midi);
            this.wasm.ms_voice_gate(voiceId, 0);
            const { ptr, len } = this.writeString('freq');
            this.wasm.ms_voice_param(voiceId, ptr, len, freq);
            this.wasm.ms_free(ptr, len);
            this.wasm.ms_voice_gate(voiceId, 1);
        } else if (!cell.sustain) {
            // Rest
            this.wasm.ms_voice_gate(voiceId, 0);
        }
        // Sustain: no-op
    }

    process(inputs, outputs, parameters) {
        if (!this.ready || !this.wasm) {
            // Output silence
            return true;
        }

        try {
            const output = outputs[0];
            if (!output || output.length === 0) return true;

            // Periodically free finished voices
            this.frameCount++;
            if (this.frameCount % 16 === 0) {
                this.wasm.ms_free_done();
            }

            // Advance sequencer
            if (this.seq) {
                const seq = this.seq;
                seq.sampleCounter += 128;
                while (seq.sampleCounter >= seq.stepDurationSamples) {
                    seq.sampleCounter -= seq.stepDurationSamples;
                    const totalSteps = seq.kind === 'shell' ? seq.steps.length : seq.cells.length;
                    if (totalSteps === 0) break;
                    const stepIdx = seq.stepIndex % totalSteps;

                    if (seq.kind === 'shell') {
                        this.seqApplyShellStep(seq, stepIdx);
                    } else {
                        this.seqApplyCompositeStep(seq, stepIdx);
                    }

                    this.port.postMessage({ type: 'stepReached', step: stepIdx });
                    seq.stepIndex++;
                }
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
        } catch (err) {
            // Throttle error messages to avoid flooding the main thread
            const now = currentTime;
            if (!this._lastErrorTime || now - this._lastErrorTime > 1) {
                this._lastErrorTime = now;
                this.port.postMessage({ type: 'error', message: 'process() exception: ' + err.message });
            }
        }

        return true;
    }
}

registerProcessor('microsynth-processor', MicrosynthProcessor);
