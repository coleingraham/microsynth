//! A voice-management layer built over `Engine`'s spawn/free primitives.
//!
//! This module holds two deliberately separate mechanisms that both happen
//! to manage voices, but never reference one another:
//!
//! - [`VoiceAllocator`] enforces a voice budget and a stealing policy for
//!   ordinary polyphonic spawning. It knows nothing about note pitch, gates,
//!   or legato behavior.
//! - [`LegatoVoice`] is a single-voice mono track that decides whether a new
//!   note ties into the currently held voice (glide, no re-attack) or
//!   retriggers it (release, then re-attack). It never asks for more than
//!   one concurrent voice, so it never consults a budget or stealing policy
//!   at all — its output cannot depend on which policy (if any) is attached
//!   to the engine.
//!
//! Keeping them apart is intentional: voice budgets/stealing and mono/legato
//! note behavior are different decisions that happen to land on the same
//! spawn API, and conflating them would make a legato run's sound depend on
//! incidental allocator state elsewhere in the engine.

use crate::curve::{GlideShape, GlideSpace};
use crate::engine::Engine;
use crate::scheduler::VoiceId;
use crate::synthdef::SynthDef;
use alloc::string::String;
use alloc::vec::Vec;

// -- Voice allocation: budgets and stealing -------------------------------

/// What to do when a voice is requested but the budget is already full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StealPolicy {
    /// Decline the new voice; the existing ones are left untouched.
    Reject,
    /// Free the longest-held active voice to make room for the new one.
    Oldest,
    /// Free the most-recently-allocated active voice to make room for the
    /// new one.
    Newest,
}

/// The outcome of requesting a voice from a [`VoiceAllocator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// There is room under the budget; proceed with the spawn.
    Admit,
    /// The budget is full; free this voice first, then proceed with the
    /// spawn.
    Steal(VoiceId),
    /// The budget is full and the policy declines to make room.
    Reject,
}

/// Enforces a voice budget and a stealing policy over `Engine`'s
/// unconditional `spawn_voice`/`free_voice` primitives.
///
/// A `VoiceAllocator` only tracks *which* `VoiceId`s it currently considers
/// active; it holds no reference to `Engine` and knows nothing about what a
/// voice sounds like. Pair it with [`Engine::spawn_voice_managed`] /
/// [`Engine::spawn_voice_on_bus_managed`] to have the budget enforced
/// automatically, or drive `request`/`record_spawn`/`record_free` directly.
#[derive(Debug, Clone)]
pub struct VoiceAllocator {
    budget: usize,
    policy: StealPolicy,
    /// Active voice IDs in spawn order (oldest first), so `Oldest` and
    /// `Newest` can be read straight off the ends of the list.
    active: Vec<VoiceId>,
}

impl VoiceAllocator {
    /// Create an allocator with room for `budget` concurrent voices,
    /// applying `policy` once that budget is exceeded.
    pub fn new(budget: usize, policy: StealPolicy) -> Self {
        VoiceAllocator {
            budget,
            policy,
            active: Vec::new(),
        }
    }

    /// The configured voice budget.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// The configured stealing policy.
    pub fn policy(&self) -> StealPolicy {
        self.policy
    }

    /// The voices this allocator currently considers active, oldest first.
    pub fn active_voices(&self) -> &[VoiceId] {
        &self.active
    }

    /// Number of voices currently considered active.
    pub fn len(&self) -> usize {
        self.active.len()
    }

    /// Whether no voices are currently considered active.
    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// Decide what must happen before a new voice can be spawned: whether
    /// there is room (`Admit`), a specific existing voice must be freed
    /// first (`Steal`), or the request should be declined (`Reject`).
    ///
    /// Does not itself mutate any bookkeeping — pair with `record_spawn` (and
    /// `record_free` for a `Steal`) once the decision has been acted on.
    pub fn request(&mut self) -> Admission {
        if self.active.len() < self.budget {
            return Admission::Admit;
        }
        match self.policy {
            StealPolicy::Reject => Admission::Reject,
            StealPolicy::Oldest => self
                .active
                .first()
                .copied()
                .map(Admission::Steal)
                .unwrap_or(Admission::Reject),
            StealPolicy::Newest => self
                .active
                .last()
                .copied()
                .map(Admission::Steal)
                .unwrap_or(Admission::Reject),
        }
    }

    /// Record that `id` has just been spawned and is now active.
    pub fn record_spawn(&mut self, id: VoiceId) {
        self.active.push(id);
    }

    /// Record that `id` is no longer active (freed directly, stolen, or
    /// finished on its own). A no-op if `id` isn't tracked.
    pub fn record_free(&mut self, id: VoiceId) {
        self.active.retain(|&v| v != id);
    }
}

// -- Mono/legato voice mode -------------------------------------------------

/// Whether moving from one note to the next on a mono voice should tie into
/// the held voice or retrigger it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    /// The next note overlaps or exactly abuts the previous one: glide to
    /// the new pitch and leave the gate open. The envelope does not
    /// re-attack.
    Tie,
    /// A gap separates the notes: release, then raise the gate again at the
    /// new pitch, producing a fresh envelope attack.
    Retrigger,
}

/// Classify the join between two notes on a mono voice from their absolute
/// sample times.
///
/// `next_on <= prev_off` (the next note starts at or before the previous one
/// ends — overlapping or exactly adjacent) is a [`Transition::Tie`];
/// anything else (a gap between them) is a [`Transition::Retrigger`].
pub fn classify_transition(prev_off: u64, next_on: u64) -> Transition {
    if next_on <= prev_off {
        Transition::Tie
    } else {
        Transition::Retrigger
    }
}

/// A monophonic, legato-capable voice.
///
/// Holds at most one live engine voice for its entire lifetime. A `note_on`
/// call either starts that voice (if none is held yet), ties into it (glide
/// pitch, gate stays open — no new envelope attack), or retriggers it (gate
/// closes and reopens at the new pitch — a fresh attack), depending only on
/// whether the gate is currently open. `LegatoVoice` never asks for a second
/// concurrent voice, so it never consults a [`VoiceAllocator`]: its behavior
/// cannot depend on whatever budget or stealing policy is in force elsewhere
/// on the engine.
pub struct LegatoVoice {
    pitch_param: String,
    portamento_secs: f32,
    /// Interpolation shape/space used for the tie-portamento glide (see
    /// `note_on`'s tie branch). Defaults to `Linear`/`Pitch` — an
    /// equal-ratio sweep, the musically conventional reading of "glide this
    /// pitch parameter" (see `crate::curve::GlideSpace::Pitch`'s doc
    /// comment). Override with `with_glide` for a different reading, e.g.
    /// `GlideSpace::Raw` for a linear-in-units sweep instead.
    portamento_shape: GlideShape,
    portamento_space: GlideSpace,
    held: Option<VoiceId>,
    gate_open: bool,
}

impl LegatoVoice {
    /// `pitch_param` is the name of the `SynthDef` parameter driven by note
    /// pitch (for example `"freq"`). `portamento_secs` is the glide time
    /// used when tying two notes together, with the tie glide itself
    /// defaulting to `GlideShape::Linear`/`GlideSpace::Pitch` (see
    /// `with_glide` to configure a different shape/space).
    pub fn new(pitch_param: impl Into<String>, portamento_secs: f32) -> Self {
        LegatoVoice {
            pitch_param: pitch_param.into(),
            portamento_secs,
            portamento_shape: GlideShape::Linear,
            portamento_space: GlideSpace::Pitch,
            held: None,
            gate_open: false,
        }
    }

    /// Configure the interpolation shape/space used for the tie-portamento
    /// glide (see `note_on`'s tie branch), overriding the `Linear`/`Pitch`
    /// default.
    pub fn with_glide(mut self, shape: GlideShape, space: GlideSpace) -> Self {
        self.portamento_shape = shape;
        self.portamento_space = space;
        self
    }

    /// The voice held by this track, if any note has been started yet.
    pub fn voice(&self) -> Option<VoiceId> {
        self.held
    }

    /// Whether the held voice's gate is currently open (a note is sounding
    /// or sustaining, as opposed to idle or releasing).
    pub fn is_gate_open(&self) -> bool {
        self.gate_open
    }

    /// Start or continue a note at `pitch`.
    ///
    /// - No voice held yet: spawns one via `def`, sets its pitch, and opens
    ///   the gate — a fresh attack.
    /// - A voice is held and its gate is currently closed (a `note_off`
    ///   happened since the last note): reopens the gate at the new pitch on
    ///   that *same* voice — a retrigger, producing a fresh attack. This only
    ///   takes effect once at least one block has rendered since `note_off`,
    ///   so the closed gate value is actually observed before it reopens.
    /// - A voice is held and its gate is still open (no `note_off` yet —
    ///   overlapping or back-to-back notes): glides to the new pitch and
    ///   leaves the gate open — a legato tie, with no new attack.
    ///
    /// A held voice can disappear on its own between notes — for example a
    /// host reaping finished voices with `Engine::free_done_synths` after a
    /// full release. If the held voice no longer exists in `engine`, this is
    /// treated the same as no voice being held at all: a fresh one is
    /// spawned. The prior note had already released, so a new attack is the
    /// musically correct outcome, not just a fallback — the alternative is
    /// silently discarding the note.
    ///
    /// Returns the (possibly newly spawned) held voice.
    pub fn note_on(&mut self, engine: &mut Engine, def: &SynthDef, pitch: f32) -> VoiceId {
        if let Some(voice) = self.held
            && engine.voice_synth(voice).is_none()
        {
            self.held = None;
            self.gate_open = false;
        }

        match self.held {
            None => {
                let voice = engine.spawn_voice(def);
                let pitch_set = engine.set_voice_param(voice, &self.pitch_param, pitch);
                let gate_set = engine.set_voice_param(voice, "gate", 1.0);
                // `voice` was just spawned from `def`, so both params must
                // exist unless `pitch_param`'s name doesn't match anything in
                // `def` — a caller configuration error, not a race with the
                // engine. Surfacing it here (rather than discarding the
                // bools) is what would have caught the reap-and-silently-
                // no-op failure mode this method now guards against above.
                debug_assert!(
                    pitch_set && gate_set,
                    "freshly spawned voice {voice:?} is missing its pitch \
                     (\"{}\") or gate param — check the SynthDef",
                    self.pitch_param
                );
                self.held = Some(voice);
                self.gate_open = true;
                voice
            }
            Some(voice) if !self.gate_open => {
                let pitch_set = engine.set_voice_param(voice, &self.pitch_param, pitch);
                let gate_set = engine.set_voice_param(voice, "gate", 1.0);
                debug_assert!(
                    pitch_set && gate_set,
                    "retriggering voice {voice:?}, confirmed live just above, \
                     is missing its pitch (\"{}\") or gate param — check the \
                     SynthDef",
                    self.pitch_param
                );
                self.gate_open = true;
                voice
            }
            Some(voice) => {
                let glide_set = engine.set_voice_param_glide(
                    voice,
                    &self.pitch_param,
                    pitch,
                    self.portamento_secs,
                    self.portamento_shape,
                    self.portamento_space,
                );
                debug_assert!(
                    glide_set,
                    "tying voice {voice:?}, confirmed live just above, is \
                     missing its pitch param (\"{}\") — check the SynthDef",
                    self.pitch_param
                );
                voice
            }
        }
    }

    /// Close the held voice's gate, beginning its release stage.
    ///
    /// Does not free or despawn the voice: it stays held for the life of
    /// this track so a following `note_on` can tie or retrigger on it. Once
    /// the track itself is done, free the voice with `Engine::free_voice` or
    /// `Engine::free_done_synths`.
    pub fn note_off(&mut self, engine: &mut Engine) {
        if let Some(voice) = self.held {
            engine.set_voice_param(voice, "gate", 0.0);
            self.gate_open = false;
        }
    }
}
