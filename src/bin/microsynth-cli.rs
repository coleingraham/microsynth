use std::io::{self, Read, Write};

use clap::{Parser, Subcommand, ValueEnum};
use microsynth::dsl::{self, UGenRegistry};
use microsynth::engine::{Engine, EngineConfig};
use microsynth::ugens::register_builtins;

#[derive(Parser)]
#[command(name = "microsynth-cli", about = "Offline rendering for microsynth DSL")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile DSL from stdin and render audio offline.
    Render {
        /// Duration in seconds.
        #[arg(long, default_value = "1.0")]
        duration: f32,

        /// Sample rate in Hz.
        #[arg(long, default_value = "44100")]
        sample_rate: f32,

        /// Output format.
        #[arg(long, default_value = "json")]
        format: OutputFormat,

        /// Output file path (default: stdout for json).
        #[arg(long)]
        output: Option<String>,

        /// Which synthdef to render (defaults to first).
        #[arg(long)]
        synthdef: Option<String>,

        /// Parameter overrides (name=value, repeatable).
        #[arg(long = "param", value_parser = parse_param)]
        params: Vec<(String, f32)>,
    },
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Json,
    Wav,
}

fn parse_param(s: &str) -> Result<(String, f32), String> {
    let (name, val) = s
        .split_once('=')
        .ok_or_else(|| format!("expected name=value, got '{s}'"))?;
    let v: f32 = val
        .parse()
        .map_err(|_| format!("invalid float value: '{val}'"))?;
    Ok((name.to_string(), v))
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Render {
            duration,
            sample_rate,
            format,
            output,
            synthdef,
            params,
        } => {
            // Read DSL from stdin
            let mut source = String::new();
            io::stdin()
                .read_to_string(&mut source)
                .expect("failed to read stdin");

            // Compile
            let mut registry = UGenRegistry::new();
            register_builtins(&mut registry);

            let defs = match dsl::compile(&source, &registry) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Compile error: {e}");
                    std::process::exit(1);
                }
            };

            if defs.is_empty() {
                eprintln!("No synthdef found in input");
                std::process::exit(1);
            }

            // Select synthdef
            let def = if let Some(ref name) = synthdef {
                defs.iter()
                    .find(|d| d.name() == name)
                    .unwrap_or_else(|| {
                        eprintln!("SynthDef '{name}' not found. Available: {:?}",
                            defs.iter().map(|d| d.name()).collect::<Vec<_>>());
                        std::process::exit(1);
                    })
            } else {
                &defs[0]
            };

            // Set up engine
            let block_size = 64;
            let config = EngineConfig {
                sample_rate,
                block_size,
            };
            let mut engine = Engine::new(config);
            let synth = engine.instantiate_synthdef(def);
            engine.graph_mut().set_sink(synth.output_node());

            // Auto-set gate=1.0 for sustaining synths unless overridden
            let has_gate_param = def
                .param_names()
                .iter()
                .any(|(name, _, _)| name == "gate");
            let gate_overridden = params.iter().any(|(name, _)| name == "gate");
            if has_gate_param && !gate_overridden {
                engine.set_param(&synth, "gate", 1.0);
            }

            // Apply param overrides
            for (name, value) in &params {
                if !engine.set_param(&synth, name, *value) {
                    eprintln!("Warning: parameter '{name}' not found on synthdef '{}'", def.name());
                }
            }

            engine.prepare();

            // Render
            let num_blocks = ((duration * sample_rate) / block_size as f32).ceil() as usize;
            let channels = engine.render_offline(num_blocks);
            let num_samples = channels.first().map_or(0, |c| c.len());

            match format {
                OutputFormat::Json => {
                    write_json(&channels, sample_rate, block_size, num_samples, &output);
                }
                OutputFormat::Wav => {
                    let path = output.unwrap_or_else(|| {
                        eprintln!("--output is required for wav format");
                        std::process::exit(1);
                    });
                    write_wav(&channels, sample_rate, &path);
                }
            }
        }
    }
}

fn write_json(
    channels: &[Vec<f32>],
    sample_rate: f32,
    block_size: usize,
    num_samples: usize,
    output: &Option<String>,
) {
    // Build JSON manually to avoid adding serde as a dependency.
    let mut json = String::new();
    json.push_str(&format!(
        "{{\"sample_rate\":{},\"block_size\":{},\"num_samples\":{},\"channels\":[",
        sample_rate as u32, block_size, num_samples
    ));
    for (i, ch) in channels.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push('[');
        for (j, &s) in ch.iter().enumerate() {
            if j > 0 {
                json.push(',');
            }
            json.push_str(&format!("{s}"));
        }
        json.push(']');
    }
    json.push_str("]}");

    if let Some(path) = output {
        std::fs::write(path, &json).expect("failed to write output file");
    } else {
        io::stdout()
            .write_all(json.as_bytes())
            .expect("failed to write stdout");
    }
}

fn write_wav(channels: &[Vec<f32>], sample_rate: f32, path: &str) {
    let num_channels = channels.len() as u16;
    let num_samples = channels.first().map_or(0, |c| c.len());
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate as u32 * num_channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = num_channels * (bits_per_sample / 8);
    let data_size = num_samples as u32 * num_channels as u32 * (bits_per_sample as u32 / 8);
    let file_size = 36 + data_size;

    let mut buf: Vec<u8> = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&(sample_rate as u32).to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    // Interleave channels and convert to 16-bit PCM
    for i in 0..num_samples {
        for ch in channels {
            let sample = ch.get(i).copied().unwrap_or(0.0);
            let clamped = sample.clamp(-1.0, 1.0);
            let pcm = (clamped * 32767.0) as i16;
            buf.extend_from_slice(&pcm.to_le_bytes());
        }
    }

    std::fs::write(path, &buf).expect("failed to write WAV file");
}
