//! Scheduling a sequence of musical-time parameter updates in one call.
//!
//! [`crate::musical_time`] converts musical positions and glide lengths to
//! sample-time terms as pure math, with no dependency on the audio graph.
//! This module is the thin, non-pure layer on top of that conversion: it
//! takes a sequence of [`MusicalGlideSegment`]s, converts them via a
//! [`TimeConfig`], and schedules the resulting events on a [`Scheduler`] —
//! so a caller can hand over a whole sequence of tempo-relative updates in
//! one call instead of converting and scheduling each one by hand.

use crate::musical_time::{MusicalGlideSegment, TimeConfig};
use crate::scheduler::{Scheduler, VoiceId};

/// Schedule a sequence of musical-time parameter updates on `scheduler`, for
/// `voice`'s `param`, in one call.
///
/// Each segment's position and glide length are converted to sample-time
/// terms via `config` (the pure conversion — see
/// [`TimeConfig::sequence_to_samples`]), then scheduled with
/// [`Scheduler::schedule_param_glide`] in the order given, carrying that
/// segment's own shape and space through unchanged. `config`'s conversion is
/// pure and deterministic, so identical `segments` and `config` always
/// produce an identical stream of scheduled events, and rendering the same
/// `segments` at a different tempo (a different `bpm` on `config`) scales
/// every position and every glide length proportionally.
///
/// Each segment's `position` is where its glide starts, not where it ends,
/// and neither this function nor the conversion it calls validates that a
/// segment's shape or glide length are musically sensible — see
/// [`MusicalGlideSegment`] for both notes.
pub fn schedule_musical_glides(
    scheduler: &mut Scheduler,
    config: &TimeConfig,
    voice: VoiceId,
    param: &str,
    segments: &[MusicalGlideSegment],
) {
    for event in config.sequence_to_samples(segments) {
        scheduler.schedule_param_glide(
            event.time,
            voice,
            param,
            event.target,
            event.glide_secs,
            event.shape,
            event.space,
        );
    }
}
