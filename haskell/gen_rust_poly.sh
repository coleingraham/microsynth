#!/usr/bin/env bash
# Emit a Rust-DSL `poly` patch with N independent filtered-saw voices, matching
# Microsynth.Demo.polyVoices (55 Hz chromatic cluster, cutoff = freq*6).
# Usage: bash gen_rust_poly.sh N > poly_N.synth
set -eu
N=${1:?usage: gen_rust_poly.sh N}
echo "synthdef poly amp=0.1 ="
awk -v n="$N" 'BEGIN{
  for (i = 0; i < n; i++) {
    f = 55 * (2 ^ (i / 12)); c = f * 6;
    printf "  let v%d = lpf (saw %.4f) %.4f 1.5 * perc 0.01 0.6\n", i, f, c;
  }
  printf "  (";
  for (i = 0; i < n; i++) printf "%sv%d", (i ? " + " : ""), i;
  printf ") * amp\n";
}'
