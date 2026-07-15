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
renderHash :: SynthDef -> Map.Map ParamName Sample -> Word64
renderHash sdef overrides =
  foldl step 1469598103934665603 (concatMap VU.toList chans)
  where
    chans = renderOffline sdef (SampleRate 44100) (SampleCount 44100) overrides
    step acc x = acc * 1099511628211 + fromIntegral (castFloatToWord32 (unSample x))

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

  -- Golden byte output: every reference render must hash to its pinned value.
  -- These lock the exact samples so any behaviour-changing refactor is caught.
  describe "golden render hashes (byte-exact output)" $ do
    it "tone matches its golden hash" $
      renderHash tone Map.empty `shouldBe` 8837859338374538051
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

    it "exposes the Lpf node's ports as named binding roles" $ do
      let mlpf = find ((== "Lpf") . nodeTag) nodes
      fmap (map fst . nodePorts) mlpf `shouldBe` Just ["sig", "cutoff", "q"]
      fmap nodeArity mlpf `shouldBe` Just 3

    it "gives leaves (Param/Const) no ports" $ do
      let leaves = filter (\nd -> nodeTag nd `elem` ["Param", "Const"]) nodes
      map nodePorts leaves `shouldSatisfy` all null
      map nodeArity leaves `shouldSatisfy` all (== 0)

  -- Rebuilding a SynthDef from its flat node list (the proposer's entry point)
  -- recovers the same params and renders byte-identically.
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
