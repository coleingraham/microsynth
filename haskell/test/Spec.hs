-- | Numeric sanity checks for the render path.
module Main (main) where

import Data.Word (Word64)
import GHC.Float (castFloatToWord32)
import qualified Data.Map.Strict as Map
import qualified Data.Vector.Unboxed as VU
import Test.Hspec

import Microsynth

-- | A deterministic FNV-1a hash over the raw 32-bit bit patterns of every
-- rendered sample. This is the golden safety net: it pins the exact byte output
-- of the reference renders so any refactor that changes a single sample fails
-- loudly. The constants below were captured from the pre-refactor engine.
renderHash :: SynthDef -> Map.Map String Float -> Word64
renderHash sdef overrides =
  foldl step 1469598103934665603 (concatMap VU.toList chans)
  where
    chans = renderOffline sdef 44100 44100 overrides
    step acc x = acc * 1099511628211 + fromIntegral (castFloatToWord32 x)

-- | A bare 440 Hz sine at unit amplitude.
sine :: SynthDef
sine = synthdef "sine" $ do
  freq <- param "freq" 440
  out (sinOsc freq 0)

main :: IO ()
main = hspec $ do
  describe "renderOffline (sinOsc 440)" $ do
    let sr    = 44100
        n     = sr
        chans = renderOffline sine (fromIntegral sr) n Map.empty
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
      let sr   = 44100 :: Int
          n    = 2048
          base = head (renderOffline sine (fromIntegral sr) n Map.empty)
          hi   = head (renderOffline sine (fromIntegral sr) n
                         (Map.fromList [("freq", 880)]))
      (base == hi) `shouldBe` False

  describe "demo synthdefs compile and render" $
    it "renders the filtered percussive demo without error" $ do
      let v = head (renderOffline demo 44100 44100 Map.empty)
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
