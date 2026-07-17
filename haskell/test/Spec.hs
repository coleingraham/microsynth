{-# LANGUAGE OverloadedStrings #-}

-- | Numeric sanity checks for the render path.
module Main (main) where

import Data.List (find)
import Data.Word (Word64)
import GHC.Float (castFloatToWord32)
import qualified Data.Map.Strict as Map
import qualified Data.Vector.Unboxed as VU
import Test.Hspec

import Microsynth
import Microsynth.SynthDef.Introspect (nodeArity, nodePorts, nodeTag)
import Microsynth.SynthDef.IR
  (IR (..), IRNode (..), decodeSynthDef, encodeSynthDef, fromIR, toIR)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck (Positive (..))

-- | A deterministic FNV-1a hash over the raw 32-bit bit patterns of every
-- rendered sample. This is the golden safety net: it pins the exact byte output
-- of the reference renders so any refactor that changes a single sample fails
-- loudly. The constants below were captured from the pre-refactor engine.
--
-- __Portability.__ A bit-exact hash is only a valid golden for a patch whose
-- output does not depend on libm's per-platform rounding. @demo@\/@pad@\/@poly@
-- qualify: their only transcendentals are the biquad's @sin@\/@cos@, evaluated
-- at a single constant @w0@. A patch that evaluates @sin@ across many distinct
-- arguments (e.g. 'tone', a raw sine oscillator) does /not/ — libm results
-- differ by an ulp between platforms, so such a patch is checked against an
-- analytic reference instead (see 'toneMaxErr').
renderHash :: SynthDef -> Map.Map ParamName Sample -> Word64
renderHash sdef overrides =
  foldl step 1469598103934665603 (concatMap VU.toList chans)
  where
    chans = renderOffline sdef (SampleRate 44100) (SampleCount 44100) overrides
    step acc x = acc * 1099511628211 + fromIntegral (castFloatToWord32 (unSample x))

-- | The worst absolute deviation of a rendered 'tone' from the analytic sine it
-- is defined to be: @sin(tau * 440 * i \/ sr) * 0.5@.
--
-- This replaces a bit-exact hash for the one reference patch that is not
-- bit-reproducible across platforms. It is a stronger check in kind — it pins
-- what the patch /is/ rather than merely that it has not changed — at the cost
-- of not catching sub-tolerance drift.
--
-- The tolerance must clear the engine's own @Float@ phase-accumulator drift:
-- the phase advances by @440\/44100@ per sample with a rounding error of up to
-- ~6e-8 each, so over @n@ samples the phase can drift by @n * 6e-8@ and the
-- sine by @tau@ times that. At @n = 4410@ that bounds the error at ~1.7e-3,
-- hence the 5e-3 tolerance. Measured worst case is 6.3e-5 (the drift is a
-- random walk, not the worst case), so there is ~80x headroom — while a real
-- defect (wrong frequency, amplitude, or waveform) moves the error to ~0.5.
toneMaxErr :: Int -> Double
toneMaxErr n = maximum [ abs (rendered i - expected i) | i <- [0 .. n - 1] ]
  where
    sr         = 44100
    v          = head (renderOffline tone (SampleRate sr) (SampleCount n) Map.empty)
    rendered i = realToFrac (unSample (v VU.! i))
    expected i = sin (2 * pi * 440 * fromIntegral i / realToFrac sr) * 0.5

-- | A bare 440 Hz sine at unit amplitude.
sine :: SynthDef
sine = synthdef "sine" $ do
  freq <- param "freq" 440
  out (sinOsc freq 0)

main :: IO ()
main = hspec $ do
  describe "renderOffline (sinOsc 440)" $ do
    let n     = 44100
        chans = renderOffline sine (SampleRate 44100) (SampleCount n) Map.empty
        v     = head chans

    it "produces exactly one (mono) channel" $
      length chans `shouldBe` 1

    it "produces the requested number of samples" $
      VU.length v `shouldBe` n

    it "keeps every sample within [-1, 1]" $
      VU.all (\x -> x >= -1.0001 && x <= 1.0001) v `shouldBe` True

    it "actually oscillates (peak near full scale)" $
      VU.maximum v `shouldSatisfy` (> 0.9)

    it "is roughly zero-mean (no DC offset)" $ do
      let mean = VU.sum v / fromIntegral (VU.length v)
      abs mean `shouldSatisfy` (< 0.05)

  describe "parameter overrides" $
    it "changes the rendered signal when freq is overridden" $ do
      let n    = 2048
          base = head (renderOffline sine (SampleRate 44100) (SampleCount n) Map.empty)
          hi   = head (renderOffline sine (SampleRate 44100) (SampleCount n)
                         (Map.fromList [("freq", 880)]))
      (base == hi) `shouldBe` False

  describe "demo synthdefs compile and render" $
    it "renders the filtered percussive demo without error" $ do
      let v = head (renderOffline demo (SampleRate 44100) (SampleCount 44100) Map.empty)
      VU.length v `shouldBe` 44100

  -- 'tone' is a raw sine oscillator, so its output is not bit-reproducible
  -- across platforms (libm `sin`, 44100 distinct arguments). It is pinned to
  -- the analytic sine it is defined to be instead of to a byte hash.
  describe "tone (analytic reference)" $
    it "matches an analytic 440 Hz sine within Float drift tolerance" $
      toneMaxErr 4410 `shouldSatisfy` (< 5.0e-3)

  -- Golden byte output: every reference render must hash to its pinned value.
  -- These lock the exact samples so any behaviour-changing refactor is caught.
  describe "golden render hashes (byte-exact output)" $ do
    it "demo matches its golden hash" $
      renderHash demo Map.empty `shouldBe` 13369518344239766915
    it "pad matches its golden hash" $
      renderHash pad Map.empty `shouldBe` 4387305950413733972
    it "poly (8 voices) matches its golden hash" $
      renderHash (polyVoices 8) Map.empty `shouldBe` 11557555834769524848

  -- Descriptor-derived introspection over the compiled graph.
  describe "graph introspection (Microsynth.SynthDef.Introspect)" $ do
    let nodes = sdNodes demo

    it "tags every demo node with its serialization kind" $ do
      let tags = map nodeTag nodes
      -- demo = lpf (saw freq) (freq*6) 1.5 * (perc ..) * amp
      tags `shouldContain` ["Saw"]
      tags `shouldContain` ["Lpf"]
      tags `shouldContain` ["Perc"]
      tags `shouldContain` ["Param"]
      tags `shouldContain` ["BinOp"]

    it "exposes the Lpf node's ports as named roles" $ do
      let mlpf = find ((== "Lpf") . nodeTag) nodes
      -- Names mirror the Rust spec() ports (src/ugens/filters.rs).
      fmap (map fst . nodePorts) mlpf `shouldBe` Just ["in", "freq", "q"]
      fmap nodeArity mlpf `shouldBe` Just 3

    it "gives leaves (Param/Const) no ports" $ do
      let leaves = filter (\nd -> nodeTag nd `elem` ["Param", "Const"]) nodes
      map nodePorts leaves `shouldSatisfy` all null
      map nodeArity leaves `shouldSatisfy` all (== 0)

  -- Rebuilding a SynthDef from its flat node list recovers the same params and
  -- renders byte-identically.
  describe "mkSynthDef (rebuild flat graph)" $ do
    let rebuilt = mkSynthDef (sdName demo) (sdNodes demo) (sdOutput demo)
    it "recovers the declared parameters" $
      sdParams rebuilt `shouldBe` sdParams demo
    it "renders byte-identically to the original" $
      renderHash rebuilt Map.empty `shouldBe` renderHash demo Map.empty

  -- Interchange IR: the structural map and the JSON bytes must both round-trip.
  describe "SynthDef IR round-trip (Microsynth.SynthDef.IR)" $ do
    let irRoundTrips d   = fromIR (toIR d) `shouldBe` Right d
        jsonRoundTrips d = decodeSynthDef (encodeSynthDef d) `shouldBe` Right d

    it "round-trips demo through the IR type" $ irRoundTrips demo
    it "round-trips tone through JSON bytes"  $ jsonRoundTrips tone
    it "round-trips demo through JSON bytes"  $ jsonRoundTrips demo
    it "round-trips pad through JSON bytes"   $ jsonRoundTrips pad
    it "round-trips poly (1 voice) through JSON bytes"   $ jsonRoundTrips (polyVoices 1)
    it "round-trips poly (16 voices) through JSON bytes" $ jsonRoundTrips (polyVoices 16)

    it "numbers node ids 0..n-1 as a real field" $
      map irnId (irNodes (toIR demo)) `shouldBe` map NodeId [0 .. length (sdNodes demo) - 1]

    prop "round-trips poly through JSON for any voice count" $ \(Positive k) ->
      let n = 1 + k `mod` 32
      in decodeSynthDef (encodeSynthDef (polyVoices n)) == Right (polyVoices n)
