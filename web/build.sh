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
#
# Note: Requires rustup-managed cargo (not Homebrew). If you have both,
#   run: brew uninstall rust
#   or set CARGO to the rustup-managed binary before running this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WASM_RAW="$PROJECT_ROOT/target/wasm32-unknown-unknown/release/microsynth.wasm"

# Use rustup-managed cargo if available (handles cross-compilation targets).
# Homebrew's cargo doesn't support rustup-installed targets.
if [ -z "${CARGO:-}" ]; then
    if [ -x "$HOME/.cargo/bin/cargo" ]; then
        CARGO="$HOME/.cargo/bin/cargo"
    elif [ -x "$HOME/.rustup/toolchains/stable-$(uname -m)-apple-darwin/bin/cargo" ] 2>/dev/null; then
        CARGO="$HOME/.rustup/toolchains/stable-$(uname -m)-apple-darwin/bin/cargo"
    else
        CARGO="cargo"
    fi
fi

echo "Building microsynth for wasm32-unknown-unknown..."
echo "  Using: $CARGO"
"$CARGO" build \
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
