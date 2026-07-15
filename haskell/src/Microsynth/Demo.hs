{-# LANGUAGE OverloadedStrings #-}

-- | Example SynthDefs, written directly in the embedded DSL.
--
-- These are the payoff of the Haskell version: a SynthDef is just a Haskell
-- value. @osc * env * amp@ and @freq * 6@ are real, type-checked expressions —
-- there is no lexer, parser, or compiler between this text and the graph.
module Microsynth.Demo
  ( registry
  , demo
  , tone
  , pad
  , polyVoices
  , polyFreqs
  ) where

import Data.List (foldl1')

import Microsynth.Signal
import Microsynth.SynthDef
import Microsynth.Types (Sample, SynthName)

-- | The CLI's name -> SynthDef table.
registry :: [(SynthName, SynthDef)]
registry =
  [ ("demo", demo)
  , ("tone", tone)
  , ("pad",  pad)
  ]

-- | A pure sine tone.
tone :: SynthDef
tone = synthdef "tone" $ do
  freq <- param "freq" 440
  amp  <- param "amp"  0.5
  out (sinOsc freq 0 * amp)

-- | A filtered, percussive saw — the end-to-end proof of concept.
demo :: SynthDef
demo = synthdef "demo" $ do
  freq <- param "freq" 220
  amp  <- param "amp"  0.4
  let osc = saw freq
      env = perc 0.01 0.6
  out (lpf osc (freq * 6) 1.5 * env * amp)

-- | A sustained-ish filtered saw pad (no gate; uses a long release).
pad :: SynthDef
pad = synthdef "pad" $ do
  freq <- param "freq" 110
  amp  <- param "amp"  0.3
  let osc = saw freq + saw (freq * 1.01) -- light detune
      env = perc 0.2 1.5
  out (lpf osc (freq * 4) 2 * env * amp)

-- | Frequencies for an @n@-voice chromatic cluster starting at 55 Hz. Shared
-- with the Rust patch generator so both engines render the same graph shape.
polyFreqs :: Int -> [Sample]
polyFreqs n = [ 55 * 2 ** (fromIntegral i / 12) | i <- [0 .. n - 1] ]

-- | An @n@-voice polyphonic synth: @n@ independent filtered-saw voices, each
-- with its own frequency and envelope, summed and scaled.
--
-- Note how the whole graph is generated with an ordinary fold over the
-- 'Num'-instance @(+)@ — building a 64-voice synth is a one-liner, and every
-- voice is still a distinct, stateful sub-graph.
polyVoices :: Int -> SynthDef
polyVoices n = synthdef "poly" $ do
  amp <- param "amp" 0.1
  let voice f =
        let osc = saw (constSig f)
            env = perc 0.01 0.6
        in lpf osc (constSig (f * 6)) 1.5 * env
      mix = foldl1' (+) (map voice (polyFreqs (max 1 n)))
  out (mix * amp)
