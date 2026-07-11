#!/usr/bin/env bash
# Head-to-head: render the same patch for DURATION seconds, N times each,
# for the Rust engine (both profiles, if built) and this Haskell scaffold.
#
# Build first:
#   (cd .. && cargo build --release --features std --bin microsynth-cli)
#   (cd .. && cargo build --release --features std --bin microsynth-cli \
#             --config profile.release.opt-level=3 --config profile.release.codegen-units=1 \
#             --target-dir target-speed)
#   cabal build
#
# Usage: bash bench.sh [DURATION_SECONDS=60] [RUNS=5]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
PATCH="$HERE/demo.synth"
DUR=${1:-60}
SR=44100
N=${2:-5}

RUST_SIZE="$ROOT/target/release/microsynth-cli"
RUST_SPEED="$ROOT/target-speed/release/microsynth-cli"
HS=$(cabal list-bin microsynth-cli 2>/dev/null)

run_rust() { "$1" render --duration "$DUR" --sample-rate "$SR" --format wav --output "$2" < "$PATCH" >/dev/null; }
run_hs()   { "$HS" --duration "$DUR" --sample-rate "$SR" -o "$2" >/dev/null; }

bench() {
  local name="$1" runner="$2" bin="$3"
  [ -x "$bin" ] || { printf "%-22s  (binary not built, skipped)\n" "$name"; return; }
  local out="$TMP/${name}.wav" best=999999 sum=0 t s e
  $runner "$bin" "$out"                 # warm-up
  for _ in $(seq 1 "$N"); do
    s=$(date +%s.%N); $runner "$bin" "$out"; e=$(date +%s.%N)
    t=$(echo "$e - $s" | bc -l); sum=$(echo "$sum + $t" | bc -l)
    (( $(echo "$t < $best" | bc -l) )) && best=$t
  done
  printf "%-22s  min=%7.3fs  mean=%7.3fs\n" "$name" "$best" "$(echo "scale=4; $sum/$N" | bc -l)"
}

echo "Patch: filtered percussive saw   Duration: ${DUR}s @ ${SR}Hz   Runs: $N"
echo "--------------------------------------------------------------------------"
bench "rust (opt-level=s)"      run_rust "$RUST_SIZE"
bench "rust (opt-level=3,lto)"  run_rust "$RUST_SPEED"
bench "haskell (-O2)"           run_hs   "$HS"
