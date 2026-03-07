//! Sample-accurate event scheduler for the synthesis engine.
//!
//! Events are dispatched at block boundaries: an event scheduled for sample N
//! is processed at the start of the block containing sample N.
//!
//! # Event types
//!
//! - **SpawnSynth**: Instantiate a SynthDef and connect it to a target bus/node.
//! - **SetParam**: Change a named parameter on a live synth.
//! - **FreeSynth**: Remove a synth from the graph.
//! - **SetGate**: Shorthand for setting the "gate" parameter (note-on/note-off).

use alloc::string::String;
use alloc::vec::Vec;

/// A unique handle for identifying scheduled synths across events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoiceId(pub u64);

/// An action to perform at a scheduled time.
#[derive(Clone)]
pub enum EventAction {
    /// Set a named parameter on an existing voice (instant).
    SetParam {
        voice: VoiceId,
        param: String,
        value: f32,
    },
    /// Set a named parameter with a smooth glide to the target.
    /// Used for crescendo, diminuendo, pitch bends, filter sweeps, etc.
    SetParamGlide {
        voice: VoiceId,
        param: String,
        target: f32,
        glide_secs: f32,
    },
    /// Set the gate parameter (convenience for note-on/note-off).
    /// gate > 0 = note on, gate = 0 = note off.
    SetGate {
        voice: VoiceId,
        value: f32,
    },
    /// Remove a voice from the graph.
    FreeSynth {
        voice: VoiceId,
    },
}

/// A scheduled event: an action to perform at a specific sample time.
#[derive(Clone)]
pub struct Event {
    /// The sample offset at which this event should be dispatched.
    pub time: u64,
    /// The action to perform.
    pub action: EventAction,
}

/// Priority queue of events sorted by time (earliest first).
pub struct Scheduler {
    events: Vec<Event>,
    next_voice_id: u64,
}

impl Scheduler {
    /// Create a new empty scheduler.
    pub fn new() -> Self {
        Scheduler {
            events: Vec::new(),
            next_voice_id: 1,
        }
    }

    /// Allocate a new unique VoiceId. Use this when spawning synths to
    /// get an ID you can reference in later events.
    pub fn alloc_voice_id(&mut self) -> VoiceId {
        let id = VoiceId(self.next_voice_id);
        self.next_voice_id += 1;
        id
    }

    /// Schedule an event. Events are kept sorted by time.
    pub fn schedule(&mut self, event: Event) {
        // Binary search for insertion point to maintain sorted order
        let pos = self.events.partition_point(|e| e.time <= event.time);
        self.events.insert(pos, event);
    }

    /// Schedule a parameter change at a specific sample time.
    pub fn schedule_set_param(
        &mut self,
        time: u64,
        voice: VoiceId,
        param: impl Into<String>,
        value: f32,
    ) {
        self.schedule(Event {
            time,
            action: EventAction::SetParam {
                voice,
                param: param.into(),
                value,
            },
        });
    }

    /// Schedule a gate change (note on/off) at a specific sample time.
    pub fn schedule_gate(&mut self, time: u64, voice: VoiceId, value: f32) {
        self.schedule(Event {
            time,
            action: EventAction::SetGate { voice, value },
        });
    }

    /// Schedule a parameter glide at a specific sample time.
    pub fn schedule_param_glide(
        &mut self,
        time: u64,
        voice: VoiceId,
        param: impl Into<String>,
        target: f32,
        glide_secs: f32,
    ) {
        self.schedule(Event {
            time,
            action: EventAction::SetParamGlide {
                voice,
                param: param.into(),
                target,
                glide_secs,
            },
        });
    }

    /// Schedule a synth removal at a specific sample time.
    pub fn schedule_free(&mut self, time: u64, voice: VoiceId) {
        self.schedule(Event {
            time,
            action: EventAction::FreeSynth { voice },
        });
    }

    /// Drain all events whose time falls before `deadline` (exclusive).
    /// Returns events in chronological order.
    pub fn drain_before(&mut self, deadline: u64) -> Vec<Event> {
        let split_pos = self.events.partition_point(|e| e.time < deadline);
        if split_pos == 0 {
            return Vec::new();
        }
        let remaining = self.events.split_off(split_pos);
        let drained = core::mem::replace(&mut self.events, remaining);
        drained
    }

    /// Check if there are any pending events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Number of pending events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Clear all pending events.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}
