-- | Graph utilities.
--
-- Port of the topological sort at the core of Rust @src/graph.rs@
-- (Kahn's algorithm). The engine renders nodes in this order so that every
-- input is already computed when a node runs — which is exactly what makes
-- Rust's @unsafe@ input-pointer gathering sound, and what lets us index the
-- per-node output vector safely here.
module Microsynth.Graph
  ( topoSort
  ) where

import Data.List (foldl')
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map

-- | Kahn topological sort. @topoSort n edges@ where nodes are @0..n-1@ and
-- @edges@ are @(from, to)@ dependency edges (@from@ must run before @to@).
-- Returns a valid render order. Assumes a DAG.
topoSort :: Int -> [(Int, Int)] -> [Int]
topoSort n edges = go start indeg0 []
  where
    adj :: Map Int [Int]
    adj = Map.fromListWith (++) [ (f, [t]) | (f, t) <- edges ]

    indeg0 :: Map Int Int
    indeg0 = Map.fromListWith (+)
               ([ (t, 1) | (_, t) <- edges ] ++ [ (i, 0) | i <- [0 .. n - 1] ])

    start = [ i | i <- [0 .. n - 1], Map.findWithDefault 0 i indeg0 == 0 ]

    go [] _ acc = reverse acc
    go (x : xs) indeg acc =
      let outs            = Map.findWithDefault [] x adj
          (queue', indeg') = foldl' relax (xs, indeg) outs
          relax (q, ind) t =
            let d    = Map.findWithDefault 0 t ind - 1
                ind' = Map.insert t d ind
            in (if d == 0 then q ++ [t] else q, ind')
      in go queue' indeg' (x : acc)
