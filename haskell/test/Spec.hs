-- | Numeric sanity checks for the render path.
module Main (main) where

import qualified Data.Map.Strict as Map
import qualified Data.Vector.Unboxed as VU
import Test.Hspec

import Microsynth

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
