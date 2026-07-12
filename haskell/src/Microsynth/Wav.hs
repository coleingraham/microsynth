-- | Minimal 16-bit PCM WAV writer.
--
-- Direct port of @write_wav@ in Rust @src/bin/microsynth-cli.rs@: a hand-built
-- RIFF/WAVE header followed by interleaved, clamped 16-bit samples. No audio
-- library, matching the Rust project's zero-dependency ethos.
module Microsynth.Wav
  ( writeWav
  ) where

import Data.ByteString.Builder
import qualified Data.ByteString.Lazy as BL
import Data.Int (Int16)
import qualified Data.Vector.Unboxed as VU

import Microsynth.Types (Sample, SampleRate (..))

-- | @writeWav path sampleRate channels@ writes each channel (equal length)
-- interleaved as little-endian 16-bit PCM.
writeWav :: FilePath -> SampleRate -> [VU.Vector Sample] -> IO ()
writeWav path sr channels =
  BL.writeFile path (toLazyByteString (header <> samples))
  where
    numCh      = length channels
    numSamples = case channels of
                   (c : _) -> VU.length c
                   []      -> 0
    bitsPerSample = 16 :: Int
    bytesPerSamp  = bitsPerSample `div` 8
    byteRate      = round (unSampleRate sr) * numCh * bytesPerSamp
    blockAlign    = numCh * bytesPerSamp
    dataSize      = numSamples * numCh * bytesPerSamp
    fileSize      = 36 + dataSize

    header =
         string7 "RIFF" <> word32LE (fromIntegral fileSize) <> string7 "WAVE"
      <> string7 "fmt " <> word32LE 16 <> word16LE 1
      <> word16LE (fromIntegral numCh)
      <> word32LE (round (unSampleRate sr))
      <> word32LE (fromIntegral byteRate)
      <> word16LE (fromIntegral blockAlign)
      <> word16LE (fromIntegral bitsPerSample)
      <> string7 "data" <> word32LE (fromIntegral dataSize)

    samples = mconcat
      [ int16LE (toPcm ((channels !! ch) VU.! i))
      | i  <- [0 .. numSamples - 1]
      , ch <- [0 .. numCh - 1]
      ]

toPcm :: Sample -> Int16
toPcm x = round (clamped * 32767)
  where clamped = max (-1) (min 1 x)
