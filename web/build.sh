#!/bin/bash
# Build microsynth WASM and prepare the web directory.
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

echo "Building microsynth for wasm32-unknown-unknown..."
cargo build \
    --manifest-path "$PROJECT_ROOT/Cargo.toml" \
    --target wasm32-unknown-unknown \
    --release \
    --features web \
    --no-default-features

echo "Running wasm-bindgen..."
wasm-bindgen \
    "$PROJECT_ROOT/target/wasm32-unknown-unknown/release/microsynth.wasm" \
    --out-dir "$SCRIPT_DIR/pkg" \
    --target web \
    --no-typescript

echo ""
echo "Build complete! Files in web/pkg/"
echo "To run: cd $SCRIPT_DIR && python3 -m http.server 8080"
echo "Then open http://localhost:8080"
