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

# Resolve the shared cargo target-dir (.cargo/config.toml) by asking cargo,
# rather than assuming the conventional in-repo ./target. `cargo metadata`'s
# config discovery is CWD-based, exactly like `cargo build`'s (a
# --manifest-path alone does not pick it up from a working directory outside
# the checkout) -- so this shells into PROJECT_ROOT before asking, and the
# actual build calls below pass --target-dir explicitly rather than relying
# on this script's own (possibly foreign) working directory to resolve the
# same config on their own.
TARGET_DIR="$(cd "$PROJECT_ROOT" && cargo metadata --manifest-path "$PROJECT_ROOT/Cargo.toml" \
    --format-version 1 --no-deps 2>/dev/null | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')"
if [ -z "$TARGET_DIR" ]; then
  echo "error: could not resolve the cargo target directory via 'cargo metadata'" >&2
  exit 1
fi
WASM_OUTPUT="$TARGET_DIR/wasm32-unknown-unknown/release/microsynth.wasm"

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
# 'ir' (MOT-640) is included so the wasm-ABI table-bound-node reachability
# path (`ms_compile_ir_with_tables`, src/web.rs) is actually reachable here --
# it is pure alloc+core (see Cargo.toml's `ir` feature comment), so this adds
# no new dependency, only the IR codec + compile_with_tables code itself.
echo ""
echo "==> Building raw WASM for AudioWorklet (std, ir, no wasm-bindgen)..."
"$CARGO" build \
    --manifest-path "$PROJECT_ROOT/Cargo.toml" \
    --target-dir "$TARGET_DIR" \
    --target wasm32-unknown-unknown \
    --release \
    --features std,ir \
    --no-default-features

cp "$WASM_OUTPUT" "$SCRIPT_DIR/pkg/microsynth_raw.wasm"
echo "    -> pkg/microsynth_raw.wasm"

# --- Build 2: wasm-bindgen WASM for main thread ---
# Uses 'web' feature which pulls in wasm-bindgen + js-sys, plus 'ir' (MOT-640,
# see Build 1's comment above -- same reachability path, same zero-new-deps
# rationale).
echo ""
echo "==> Building wasm-bindgen module for main thread (web, ir features)..."
"$CARGO" build \
    --manifest-path "$PROJECT_ROOT/Cargo.toml" \
    --target-dir "$TARGET_DIR" \
    --target wasm32-unknown-unknown \
    --release \
    --features web,ir \
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
