//! Integration tests for stereo pan placement at the instrument-bus stage:
//! `Engine::spawn_voice_on_bus_panned` via the raw C-ABI `ms_spawn_voice_panned`
//! (see `src/web.rs`). Mirrors `tests/wasm_raw_abi.rs`'s pattern of driving the
//! `#[no_mangle] extern "C"` exports directly.

use microsynth::web;
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn alloc_str(s: &str) -> (*mut u8, usize) {
    let len = s.len();
    let ptr = web::ms_alloc(len);
    unsafe {
        core::ptr::copy_nonoverlapping(s.as_ptr(), ptr, len);
    }
    (ptr, len)
}

fn free_str(ptr: *mut u8, len: usize) {
    unsafe { web::ms_free(ptr, len) };
}

/// Render one 128-sample stereo block through the raw C-ABI.
fn render_stereo() -> ([f32; 128], [f32; 128]) {
    let mut left = [0f32; 128];
    let mut right = [0f32; 128];
    unsafe { web::ms_render(left.as_mut_ptr(), right.as_mut_ptr()) };
    (left, right)
}

const TONE_SOURCE: &str = "synthdef test freq=440.0 = sinOsc freq 0.0";

#[test]
fn test_center_pan_matches_plain_spawn_voice_sample_for_sample() {
    // `ms_spawn_voice_panned(0.0)` must be byte-for-byte what `ms_spawn_voice`
    // produces: center pan takes the exact same direct-to-bus connection,
    // with no `Pan2` node in the path. This is what keeps a byte-exact
    // parity/golden-fixture gate elsewhere in the stack anchored to its
    // historical baseline even after stereo pan placement is wired in.
    let _guard = lock();
    unsafe {
        web::ms_init(44100.0);
        let (sptr, slen) = alloc_str(TONE_SOURCE);
        assert_eq!(web::ms_compile_def(sptr, slen), 0);
        free_str(sptr, slen);
        let voice = web::ms_spawn_voice_panned(0.0);
        assert!(voice > 0);

        let mut panned_trace = Vec::new();
        for _ in 0..10 {
            let (l, r) = render_stereo();
            panned_trace.push((l, r));
        }

        web::ms_init(44100.0);
        let (sptr2, slen2) = alloc_str(TONE_SOURCE);
        assert_eq!(web::ms_compile_def(sptr2, slen2), 0);
        free_str(sptr2, slen2);
        let voice2 = web::ms_spawn_voice();
        assert!(voice2 > 0);

        for (i, (exp_l, exp_r)) in panned_trace.iter().enumerate() {
            let (l, r) = render_stereo();
            assert_eq!(
                &l, exp_l,
                "left channel diverged from plain ms_spawn_voice at block {i}"
            );
            assert_eq!(
                &r, exp_r,
                "right channel diverged from plain ms_spawn_voice at block {i}"
            );
        }
    }
}

#[test]
fn test_nonzero_pan_makes_left_and_right_differ() {
    let _guard = lock();
    unsafe {
        web::ms_init(44100.0);
        let (sptr, slen) = alloc_str(TONE_SOURCE);
        assert_eq!(web::ms_compile_def(sptr, slen), 0);
        free_str(sptr, slen);

        // Hard right: -1.0..=1.0, so 0.8 is strongly (not maximally) right --
        // exercises the general equal-power curve, not just the pos=+-1 edge.
        let voice = web::ms_spawn_voice_panned(0.8);
        assert!(voice > 0, "expected a voice id");

        let mut left_energy = 0.0f32;
        let mut right_energy = 0.0f32;
        let mut any_sample_differs = false;
        for _ in 0..10 {
            let (l, r) = render_stereo();
            for i in 0..128 {
                left_energy += l[i].abs();
                right_energy += r[i].abs();
                if (l[i] - r[i]).abs() > 1e-6 {
                    any_sample_differs = true;
                }
            }
        }

        assert!(
            any_sample_differs,
            "expected left != right in the rendered output under a non-center pan"
        );
        assert!(
            right_energy > left_energy,
            "pan=0.8 (toward right) should favor the right channel: left={left_energy}, right={right_energy}"
        );
        // Equal-power center reference: at pos=0 each channel carries
        // cos(pi/4) = sin(pi/4) ~= 0.7071 of the mono signal. At pos=0.8 the
        // right channel's coefficient (sin(theta), theta=(pos+1)*pi/4) should
        // clearly exceed that, and the left's (cos(theta)) fall clearly under.
        let theta = (0.8f32 + 1.0) * core::f32::consts::FRAC_PI_4;
        let expected_ratio = theta.sin() / theta.cos();
        let actual_ratio = right_energy / left_energy;
        assert!(
            (actual_ratio - expected_ratio).abs() < 0.05,
            "expected right/left energy ratio ~{expected_ratio}, got {actual_ratio}"
        );
    }
}

#[test]
fn test_hard_left_pan_silences_the_right_channel() {
    let _guard = lock();
    unsafe {
        web::ms_init(44100.0);
        let (sptr, slen) = alloc_str(TONE_SOURCE);
        assert_eq!(web::ms_compile_def(sptr, slen), 0);
        free_str(sptr, slen);

        let voice = web::ms_spawn_voice_panned(-1.0);
        assert!(voice > 0);

        let mut right_peak = 0.0f32;
        let mut left_peak = 0.0f32;
        for _ in 0..10 {
            let (l, r) = render_stereo();
            for i in 0..128 {
                left_peak = left_peak.max(l[i].abs());
                right_peak = right_peak.max(r[i].abs());
            }
        }
        assert!(
            left_peak > 0.1,
            "expected audible signal on the left channel"
        );
        assert!(
            right_peak < 1e-5,
            "expected the right channel to be silent at pos=-1.0 (hard left), got peak {right_peak}"
        );
    }
}

#[test]
fn test_out_of_range_pan_is_clamped_not_rejected() {
    // Pan2 clamps its `pos` input to [-1, 1] (see `ugens::utility::Pan2::process`);
    // a caller passing an out-of-range value must not fail the spawn or panic --
    // it should behave exactly like the nearest in-range extreme.
    let _guard = lock();
    unsafe {
        web::ms_init(44100.0);
        let (sptr, slen) = alloc_str(TONE_SOURCE);
        assert_eq!(web::ms_compile_def(sptr, slen), 0);
        free_str(sptr, slen);

        let voice = web::ms_spawn_voice_panned(5.0);
        assert!(voice > 0, "an out-of-range pan must still produce a voice");

        let mut right_peak = 0.0f32;
        let mut left_peak = 0.0f32;
        for _ in 0..10 {
            let (l, r) = render_stereo();
            for i in 0..128 {
                left_peak = left_peak.max(l[i].abs());
                right_peak = right_peak.max(r[i].abs());
            }
        }
        assert!(
            right_peak > 0.1,
            "expected 5.0 to clamp to hard right and be audible there"
        );
        assert!(
            left_peak < 1e-5,
            "expected 5.0 to clamp to hard right, silencing the left channel"
        );
    }
}
