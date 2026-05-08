# 40-Pass Game Bench (TRUE 40K context growth)

Run: 2026-05-08T17:30:04.430893
Session ID: game-bench-long-1778275711
Branch: perf/memory-caps-and-adaptive-depth

## Summary

- Passes attempted: 40
- Successful: 40
- Errors: 0
- Cache hits: 39/40
- Cache misses: 1/40

## Context growth

- Pass 1 ctx: 75
- Pass 40 ctx: 38756
- Mean ctx: 19475

## TTFT

- min: 0.36s | max: 1.90s | mean: 1.54s

## Decode tok/s

- min: 25.2 | max: 55.4 | mean: 39.3

## Peak memory MB

- min: 25756 | max: 38316 | mean: 33033

## Decode cliffs (50%+ consecutive drops)

- None detected

## Per-pass detail

| # | object | ctx | ttft_s | decode | out | mem_MB | hit |
|---|---|---|---|---|---|---|---|
| 1 | Airplane | 75 | 0.36 | 42.0 | 53 | 25756 | False |
| 2 | AIFollower | 1096 | 1.06 | 42.6 | 57 | 25756 | True |
| 3 | Airport | 2118 | 1.07 | 44.4 | 31 | 25756 | True |
| 4 | Balloons | 3115 | 1.14 | 38.4 | 32 | 25756 | True |
| 5 | Birds | 4112 | 1.21 | 44.4 | 36 | 25756 | True |
| 6 | Bridges | 5113 | 1.31 | 40.3 | 35 | 25756 | True |
| 7 | Building | 6114 | 1.25 | 44.5 | 29 | 25990 | True |
| 8 | Car | 7104 | 1.26 | 42.5 | 30 | 26396 | True |
| 9 | Castle | 8095 | 1.29 | 43.2 | 31 | 26819 | True |
| 10 | Checkpoints | 9091 | 1.32 | 36.0 | 29 | 27272 | True |
| 11 | City | 10082 | 1.38 | 41.0 | 26 | 27831 | True |
| 12 | Clouds | 11070 | 1.35 | 41.2 | 33 | 28415 | True |
| 13 | Coins | 12065 | 1.39 | 48.8 | 27 | 29086 | True |
| 14 | ControlTower | 13055 | 1.41 | 39.6 | 31 | 29790 | True |
| 15 | Drone | 14048 | 1.47 | 30.6 | 29 | 30563 | True |
| 16 | ExplosionEffect | 15044 | 1.53 | 47.9 | 23 | 31411 | True |
| 17 | Fence | 16030 | 1.57 | 35.6 | 32 | 32287 | True |
| 18 | Fireworks | 17025 | 1.60 | 55.4 | 23 | 33246 | True |
| 19 | Helicopter | 18011 | 1.63 | 39.4 | 25 | 34262 | True |
| 20 | Highway | 19001 | 1.65 | 46.8 | 32 | 35321 | True |
| 21 | Hills | 19994 | 1.66 | 37.5 | 19 | 36456 | True |
| 22 | HotAirBalloon | 20977 | 1.67 | 39.0 | 28 | 36615 | True |
| 23 | House | 21969 | 1.67 | 45.3 | 23 | 36784 | True |
| 24 | Lighthouse | 22952 | 1.60 | 45.6 | 28 | 36784 | True |
| 25 | Mountains | 23942 | 1.67 | 42.9 | 28 | 36784 | True |
| 26 | Pine | 24931 | 1.69 | 36.4 | 24 | 36784 | True |
| 27 | PowerLines | 25918 | 1.71 | 31.2 | 30 | 37164 | True |
| 28 | Rain | 26911 | 1.72 | 48.7 | 27 | 37164 | True |
| 29 | Refinery | 27899 | 1.73 | 34.4 | 28 | 37164 | True |
| 30 | River | 28891 | 1.77 | 34.8 | 23 | 37164 | True |
| 31 | Road | 29874 | 1.74 | 29.1 | 28 | 37552 | True |
| 32 | RoadCones | 30865 | 1.79 | 39.8 | 28 | 37552 | True |
| 33 | Rocks | 31855 | 1.85 | 31.6 | 21 | 37552 | True |
| 34 | Runway | 32838 | 1.81 | 30.2 | 23 | 37905 | True |
| 35 | Sky | 33822 | 1.89 | 25.2 | 28 | 37905 | True |
| 36 | Stadium | 34813 | 1.87 | 38.3 | 26 | 37905 | True |
| 37 | Streetlights | 35801 | 1.90 | 36.3 | 26 | 37905 | True |
| 38 | Tank | 36787 | 1.83 | 28.5 | 22 | 38316 | True |
| 39 | Trains | 37769 | 1.83 | 37.2 | 24 | 38316 | True |
| 40 | Trees | 38756 | 1.87 | 36.8 | 31 | 38316 | True |
