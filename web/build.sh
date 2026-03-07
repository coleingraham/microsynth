#!/bin/bash
# Build microsynth WASM and prepare the web directory.
#
# Produces two WASM outputs:
#   1. pkg/microsynth.js + pkg/microsynth_bg.wasm  (wasm-bindgen, for main thread)
#   2. pkg/microsynth_raw.wasm                      (raw, for AudioWorklet)
#
# Prerequisites:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli
#
# Usage:
#   cd web && ./build.sh
#   # Then serve: python3 -m http.server 8080

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WASM_RAW="$PROJECT_ROOT/target/wasm32-unknown-unknown/release/microsynth.wasm"

# Detect a toolchain that has wasm32-unknown-unknown installed.
# Prefer stable, fall back to nightly, then default.
CARGO="cargo"
if rustup target list --toolchain stable 2>/dev/null | grep -q 'wasm32-unknown-unknown (installed)'; then
    CARGO="rustup run stable cargo"
elif rustup target list --toolchain nightly 2>/dev/null | grep -q 'wasm32-unknown-unknown (installed)'; then
    CARGO="rustup run nightly cargo"
else
    echo "Warning: wasm32-unknown-unknown not found on stable or nightly."
    echo "Run: rustup target add wasm32-unknown-unknown"
fi

echo "Building microsynth for wasm32-unknown-unknown... ($CARGO)"
$CARGO build \
    --manifest-path "$PROJECT_ROOT/Cargo.toml" \
    --target wasm32-unknown-unknown \
    --release \
    --features web \
    --no-default-features

echo "Running wasm-bindgen (main thread module)..."
wasm-bindgen \
    "$WASM_RAW" \
    --out-dir "$SCRIPT_DIR/pkg" \
    --target web \
    --no-typescript

echo "Copying raw WASM (AudioWorklet module)..."
cp "$WASM_RAW" "$SCRIPT_DIR/pkg/microsynth_raw.wasm"

echo ""
echo "Build complete! Files in web/pkg/"
ls -lh "$SCRIPT_DIR/pkg/"
echo ""
echo "To run: cd $SCRIPT_DIR && python3 -m http.server 8080"
echo "Then open http://localhost:8080"
