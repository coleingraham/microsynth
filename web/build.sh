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
WASM_OUTPUT="$PROJECT_ROOT/target/wasm32-unknown-unknown/release/microsynth.wasm"

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

echo "Using: $CARGO"
mkdir -p "$SCRIPT_DIR/pkg"

# --- Build 1: Raw WASM for AudioWorklet ---
# Uses 'std' feature (for allocator + math) but NOT 'web' (no wasm-bindgen).
# This produces a clean WASM module with only #[no_mangle] C exports.
echo ""
echo "==> Building raw WASM for AudioWorklet (std, no wasm-bindgen)..."
"$CARGO" build \
    --manifest-path "$PROJECT_ROOT/Cargo.toml" \
    --target wasm32-unknown-unknown \
    --release \
    --features std \
    --no-default-features

cp "$WASM_OUTPUT" "$SCRIPT_DIR/pkg/microsynth_raw.wasm"
echo "    -> pkg/microsynth_raw.wasm"

# --- Build 2: wasm-bindgen WASM for main thread ---
# Uses 'web' feature which pulls in wasm-bindgen + js-sys.
echo ""
echo "==> Building wasm-bindgen module for main thread (web feature)..."
"$CARGO" build \
    --manifest-path "$PROJECT_ROOT/Cargo.toml" \
    --target wasm32-unknown-unknown \
    --release \
    --features web \
    --no-default-features

echo "    Running wasm-bindgen..."
wasm-bindgen \
    "$WASM_OUTPUT" \
    --out-dir "$SCRIPT_DIR/pkg" \
    --target web \
    --no-typescript
echo "    -> pkg/microsynth.js + pkg/microsynth_bg.wasm"

echo ""
echo "Build complete! Files in web/pkg/"
ls -lh "$SCRIPT_DIR/pkg/"
echo ""
echo "To run: cd $SCRIPT_DIR && python3 -m http.server 8080"
echo "Then open http://localhost:8080"
