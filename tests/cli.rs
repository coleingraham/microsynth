//! Integration tests for the microsynth-cli binary.

use std::process::Command;

fn cli_path() -> String {
    // Use the debug binary path relative to the project root.
    let mut path = std::env::current_exe().unwrap();
    // tests are in target/debug/deps/, go up to target/debug/
    path.pop();
    path.pop();
    path.push("microsynth-cli");
    path.to_string_lossy().to_string()
}

fn run_render(dsl: &str, args: &[&str]) -> (String, String, i32) {
    let output = Command::new(cli_path())
        .arg("render")
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(dsl.as_bytes()).unwrap();
            child.wait_with_output()
        })
        .expect("failed to run microsynth-cli");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

// ============================================================================
// JSON output tests
// ============================================================================

#[test]
fn test_cli_json_output_structure() {
    let dsl = "synthdef test freq=440.0 amp=0.5 = sinOsc freq 0.0 * amp";
    let (stdout, stderr, code) = run_render(dsl, &["--duration", "0.01", "--format", "json"]);
    assert_eq!(code, 0, "CLI failed: {stderr}");

    // Parse as JSON-like structure (manual since we don't have serde in tests)
    assert!(stdout.starts_with("{\"sample_rate\":"), "unexpected JSON start: {}", &stdout[..80.min(stdout.len())]);
    assert!(stdout.contains("\"block_size\":64"));
    assert!(stdout.contains("\"num_samples\":"));
    assert!(stdout.contains("\"channels\":[["));
}

#[test]
fn test_cli_sine_produces_nonzero_samples() {
    let dsl = "synthdef test freq=440.0 amp=1.0 = sinOsc freq 0.0 * amp";
    let (stdout, _stderr, code) = run_render(dsl, &["--duration", "0.01", "--format", "json"]);
    assert_eq!(code, 0);

    // Extract first few samples from the JSON channels array
    let channels_start = stdout.find("\"channels\":[[").unwrap() + "\"channels\":[[".len();
    let first_comma = stdout[channels_start..].find(',').unwrap();
    let first_sample: f32 = stdout[channels_start..channels_start + first_comma].parse().unwrap();
    // First sample of sinOsc at phase 0 should be 0 (or very close)
    assert!(first_sample.abs() < 0.01, "first sample should be near 0, got {first_sample}");

    // But there should be non-zero samples (the sine is oscillating)
    // Check that the output contains values above 0.5
    assert!(stdout.contains("0.9") || stdout.contains("1.0") || stdout.contains("0.8"),
        "sine wave with amp=1.0 should have samples near 1.0");
}

#[test]
fn test_cli_param_override() {
    // Render with amp=0 — should produce silence
    let dsl = "synthdef test freq=440.0 amp=0.5 = sinOsc freq 0.0 * amp";
    let (stdout, _stderr, code) = run_render(dsl, &[
        "--duration", "0.01",
        "--format", "json",
        "--param", "amp=0.0",
    ]);
    assert_eq!(code, 0);

    // All samples should be 0
    let channels_start = stdout.find("\"channels\":[[").unwrap() + "\"channels\":[[".len();
    let channels_end = stdout.find("]]").unwrap();
    let samples_str = &stdout[channels_start..channels_end];
    let all_zero = samples_str.split(',').all(|s| {
        let v: f32 = s.trim().parse().unwrap_or(999.0);
        v.abs() < 1e-10
    });
    assert!(all_zero, "all samples should be 0 when amp=0");
}

#[test]
fn test_cli_gate_auto_set() {
    // A sustaining synth with gate param should auto-set gate=1.0
    let dsl = "synthdef pad freq=440.0 gate=1.0 amp=0.5 = sinOsc freq 0.0 * asr gate 0.01 0.1 * amp";
    let (stdout, _stderr, code) = run_render(dsl, &["--duration", "0.05", "--format", "json"]);
    assert_eq!(code, 0);

    // Should have non-zero samples (gate is auto-opened)
    assert!(!stdout.contains("\"channels\":[[0,0,0,0,0,0,0,0,0,0"),
        "sustaining synth should produce output with auto gate=1.0");
}

#[test]
fn test_cli_wav_output() {
    let dsl = "synthdef test freq=440.0 amp=0.5 = sinOsc freq 0.0 * amp";
    let tmp = std::env::temp_dir().join("microsynth_test_output.wav");
    let tmp_str = tmp.to_string_lossy().to_string();

    let (_stdout, stderr, code) = run_render(dsl, &[
        "--duration", "0.1",
        "--format", "wav",
        "--output", &tmp_str,
    ]);
    assert_eq!(code, 0, "CLI failed: {stderr}");

    // Verify WAV file exists and has correct header
    let data = std::fs::read(&tmp).expect("WAV file should exist");
    assert!(data.len() > 44, "WAV file too small");
    assert_eq!(&data[0..4], b"RIFF");
    assert_eq!(&data[8..12], b"WAVE");
    assert_eq!(&data[12..16], b"fmt ");
    assert_eq!(&data[36..40], b"data");

    // Check sample rate (bytes 24-27, little-endian u32)
    let sr = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    assert_eq!(sr, 44100);

    // Check channels (bytes 22-23, little-endian u16)
    let channels = u16::from_le_bytes([data[22], data[23]]);
    assert_eq!(channels, 1);

    // Check bits per sample (bytes 34-35)
    let bps = u16::from_le_bytes([data[34], data[35]]);
    assert_eq!(bps, 16);

    // Clean up
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_cli_custom_sample_rate() {
    let dsl = "synthdef test freq=440.0 amp=0.5 = sinOsc freq 0.0 * amp";
    let (stdout, _stderr, code) = run_render(dsl, &[
        "--duration", "0.01",
        "--format", "json",
        "--sample-rate", "22050",
    ]);
    assert_eq!(code, 0);
    assert!(stdout.contains("\"sample_rate\":22050"));
}

#[test]
fn test_cli_multiple_synthdefs_select() {
    let dsl = "\
synthdef first freq=440.0 amp=0.5 = sinOsc freq 0.0 * amp
synthdef second freq=880.0 amp=0.3 = saw freq * amp";
    // Select the second synthdef by name
    let (stdout, _stderr, code) = run_render(dsl, &[
        "--duration", "0.01",
        "--format", "json",
        "--synthdef", "second",
    ]);
    assert_eq!(code, 0);
    // Should have rendered successfully
    assert!(stdout.contains("\"channels\":[["));
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
fn test_cli_compile_error() {
    let dsl = "synthdef test = unknownUGen 440.0";
    let (_stdout, stderr, code) = run_render(dsl, &["--duration", "0.01"]);
    assert_ne!(code, 0, "should fail on unknown UGen");
    assert!(stderr.contains("Compile error") || stderr.contains("error"),
        "stderr should contain error info: {stderr}");
}

#[test]
fn test_cli_empty_input() {
    let (_stdout, stderr, code) = run_render("", &["--duration", "0.01"]);
    assert_ne!(code, 0, "should fail on empty input");
    assert!(!stderr.is_empty(), "stderr should have an error message");
}

#[test]
fn test_cli_unknown_synthdef_name() {
    let dsl = "synthdef test freq=440.0 = sinOsc freq 0.0";
    let (_stdout, stderr, code) = run_render(dsl, &[
        "--duration", "0.01",
        "--synthdef", "nonexistent",
    ]);
    assert_ne!(code, 0, "should fail when named synthdef not found");
    assert!(stderr.contains("not found"), "stderr should mention not found: {stderr}");
}

#[test]
fn test_cli_unknown_param_warning() {
    let dsl = "synthdef test freq=440.0 = sinOsc freq 0.0";
    let (_stdout, stderr, code) = run_render(dsl, &[
        "--duration", "0.01",
        "--param", "bogus=1.0",
    ]);
    // Should succeed but warn about unknown param
    assert_eq!(code, 0);
    assert!(stderr.contains("Warning") && stderr.contains("bogus"),
        "should warn about unknown param: {stderr}");
}
