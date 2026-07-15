-- | Native CLI: render a demo SynthDef offline to a WAV file.
--
-- The Haskell analogue of Rust @src/bin/microsynth-cli.rs@ (using
-- @optparse-applicative@ instead of @clap@). The Rust CLI reads a DSL program
-- from stdin; here SynthDefs are Haskell values selected by name from
-- 'Microsynth.Demo.registry'.
module Main (main) where

import qualified Data.Map.Strict as Map
import Options.Applicative

import Microsynth

data Opts = Opts
  { optDuration   :: Double
  , optSampleRate :: Double
  , optOutput     :: FilePath
  , optSynth      :: String
  , optVoices     :: Int
  , optParams     :: [(ParamName, Sample)]
  }

optsParser :: Parser Opts
optsParser = Opts
  <$> option auto
        ( long "duration" <> short 'd' <> metavar "SECONDS"
       <> value 2.0 <> showDefault <> help "Render duration in seconds" )
  <*> option auto
        ( long "sample-rate" <> short 'r' <> metavar "HZ"
       <> value 44100 <> showDefault <> help "Sample rate" )
  <*> strOption
        ( long "output" <> short 'o' <> metavar "FILE"
       <> value "out.wav" <> showDefault <> help "Output WAV path" )
  <*> strOption
        ( long "synthdef" <> short 's' <> metavar "NAME"
       <> value "demo" <> showDefault <> help "Which demo synthdef to render" )
  <*> option auto
        ( long "voices" <> metavar "N"
       <> value 8 <> showDefault <> help "Voice count for the 'poly' synthdef" )
  <*> many
        ( option (eitherReader parseParam)
            ( long "param" <> short 'p' <> metavar "NAME=VALUE"
           <> help "Override a parameter default (repeatable)" ) )

parseParam :: String -> Either String (ParamName, Sample)
parseParam s = case break (== '=') s of
  (name, '=' : val) -> case reads val of
    [(v, "")] -> Right (ParamName name, Sample v)
    _         -> Left ("invalid number in --param: " ++ s)
  _ -> Left ("expected NAME=VALUE in --param: " ++ s)

main :: IO ()
main = do
  opts <- execParser $ info (optsParser <**> helper)
    ( fullDesc <> progDesc "Render a microsynth demo synthdef to a WAV file" )
  let selected = case optSynth opts of
        "poly" -> Just (polyVoices (optVoices opts))
        name   -> lookup (SynthName name) registry
  case selected of
    Nothing ->
      putStrLn $ "unknown synthdef '" ++ optSynth opts
        ++ "'; available: " ++ unwords (map (unSynthName . fst) registry)
    Just sdef -> do
      let sr         = SampleRate (realToFrac (optSampleRate opts))
          numSamples = SampleCount (round (optDuration opts * optSampleRate opts))
          overrides  = Map.fromList (optParams opts)
          channels   = renderOffline sdef sr numSamples overrides
      writeWav (optOutput opts) sr channels
      putStrLn $ "Rendered '" ++ optSynth opts ++ "' ("
        ++ show (unSampleCount numSamples) ++ " samples @ " ++ show (optSampleRate opts)
        ++ " Hz) -> " ++ optOutput opts
