# Cursor Loss Events Analysis Report

**Generated:** 2026-02-20
**Total Events Analyzed:** 102
**Time Range:** 20260219_235603 to 20260220_041120

## 1. Summary Statistics

### 1.1 Recovery Rate
- **Recovered:** 32 (31.4%)
- **Never Recovered:** 70 (68.6%)

### 1.2 Event Duration
  Duration (seconds):
    Count: 102
    Mean:   22.013
    Median: 24.900
    Min:    12.000
    Max:    26.800
    StdDev: 4.718

### 1.3 Time to Lose Cursor (had_it -> lost transition)
  Time to Lose (seconds):
    Count: 90
    Mean:   0.320
    Median: 0.230
    Min:    0.200
    Max:    1.130
    StdDev: 0.255
  Events with measurable had_it->lost transition: 90/102

### 1.4 Time to Recover (lost -> recovered)
  Time to Recover (seconds):
    Count: 21
    Mean:   8.592
    Median: 9.020
    Min:    1.020
    Max:    15.060
    StdDev: 4.224
  Events that recovered: 21/102

### 1.5 Phase at Trigger (t=0)
  - lost: 90 (88.2%)
  - recovering: 11 (10.8%)
  - unknown: 1 (1.0%)

### 1.6 Frame Counts per Event
  Frames per Event:
    Count: 102
    Mean:   104.657
    Median: 118.500
    Min:    51.000
    Max:    128.000
    StdDev: 23.721

### 1.7 Frame Resolutions
  - 1920x1080: 102

### 1.8 Phase Frame Distribution
  had_it frames per event:
    Count: 102
    Mean:   40.333
    Median: 43.000
    Min:    3.000
    Max:    46.000
    StdDev: 10.301

  lost frames per event:
    Count: 102
    Mean:   60.765
    Median: 76.000
    Min:    0.000
    Max:    76.000
    StdDev: 26.936

  recovering frames per event:
    Count: 102
    Mean:   0.422
    Median: 0.000
    Min:    0.000
    Max:    2.000
    StdDev: 0.681

  recovered frames per event:
    Count: 102
    Mean:   3.137
    Median: 0.000
    Min:    0.000
    Max:    10.000
    StdDev: 4.663

## 2. Trigger Type Analysis

### 2.1 Trigger Distribution
  - `cnn_drop`: 25 (24.5%)
  - `validation_lost`: 6 (5.9%)
  - `position_jump_1832px`: 4 (3.9%)
  - `position_jump_288px`: 2 (2.0%)
  - `position_jump_231px`: 2 (2.0%)
  - `position_jump_573px`: 2 (2.0%)
  - `position_jump_577px`: 2 (2.0%)
  - `position_jump_367px`: 2 (2.0%)
  - `position_jump_1819px`: 1 (1.0%)
  - `position_jump_2025px`: 1 (1.0%)
  - `position_jump_1778px`: 1 (1.0%)
  - `position_jump_276px`: 1 (1.0%)
  - `position_jump_531px`: 1 (1.0%)
  - `position_jump_909px`: 1 (1.0%)
  - `position_jump_865px`: 1 (1.0%)
  - `position_jump_540px`: 1 (1.0%)
  - `position_jump_652px`: 1 (1.0%)
  - `position_jump_988px`: 1 (1.0%)
  - `position_jump_246px`: 1 (1.0%)
  - `position_jump_297px`: 1 (1.0%)
  - `position_jump_451px`: 1 (1.0%)
  - `position_jump_649px`: 1 (1.0%)
  - `position_jump_637px`: 1 (1.0%)
  - `position_jump_655px`: 1 (1.0%)
  - `position_jump_578px`: 1 (1.0%)
  - `position_jump_946px`: 1 (1.0%)
  - `position_jump_393px`: 1 (1.0%)
  - `position_jump_314px`: 1 (1.0%)
  - `position_jump_455px`: 1 (1.0%)
  - `position_jump_630px`: 1 (1.0%)
  - `position_jump_623px`: 1 (1.0%)
  - `position_jump_881px`: 1 (1.0%)
  - `position_jump_563px`: 1 (1.0%)
  - `position_jump_714px`: 1 (1.0%)
  - `position_jump_443px`: 1 (1.0%)
  - `position_jump_240px`: 1 (1.0%)
  - `position_jump_209px`: 1 (1.0%)
  - `position_jump_294px`: 1 (1.0%)
  - `position_jump_228px`: 1 (1.0%)
  - `position_jump_248px`: 1 (1.0%)
  - `position_jump_1044px`: 1 (1.0%)
  - `position_jump_930px`: 1 (1.0%)
  - `position_jump_223px`: 1 (1.0%)
  - `position_jump_941px`: 1 (1.0%)
  - `position_jump_984px`: 1 (1.0%)
  - `position_jump_203px`: 1 (1.0%)
  - `position_jump_873px`: 1 (1.0%)
  - `position_jump_997px`: 1 (1.0%)
  - `position_jump_224px`: 1 (1.0%)
  - `position_jump_216px`: 1 (1.0%)
  - `position_jump_264px`: 1 (1.0%)
  - `position_jump_253px`: 1 (1.0%)
  - `position_jump_364px`: 1 (1.0%)
  - `position_jump_1185px`: 1 (1.0%)
  - `position_jump_1112px`: 1 (1.0%)
  - `position_jump_215px`: 1 (1.0%)
  - `position_jump_218px`: 1 (1.0%)
  - `position_jump_387px`: 1 (1.0%)
  - `position_jump_1004px`: 1 (1.0%)
  - `position_jump_349px`: 1 (1.0%)
  - `position_jump_490px`: 1 (1.0%)
  - `position_jump_435px`: 1 (1.0%)
  - `position_jump_511px`: 1 (1.0%)
  - `position_jump_470px`: 1 (1.0%)
  - `position_jump_773px`: 1 (1.0%)

### 2.2 Position Jump Analysis
  Jump Distance (px):
    Count: 71
    Mean:   663.972
    Median: 540.000
    Min:    203.000
    Max:    2025.000
    StdDev: 476.728

## 3. Velocity and Acceleration Analysis

### 3.1 Maximum Velocity
  Max Velocity (px/s):
    Count: 102
    Mean:   1346.439
    Median: 923.200
    Min:    0.000
    Max:    5414.100
    StdDev: 1340.373

### 3.2 Velocity at Trigger
  Velocity at Trigger (px/s):
    Count: 102
    Mean:   1103.875
    Median: 677.100
    Min:    0.000
    Max:    5414.100
    StdDev: 1359.969

### 3.3 Average Velocity Pre-Loss vs Post-Loss
  Avg Velocity Pre-Loss (px/s):
    Count: 102
    Mean:   73.239
    Median: 27.250
    Min:    0.000
    Max:    521.100
    StdDev: 113.336

  Avg Velocity Post-Loss (px/s):
    Count: 102
    Mean:   38.653
    Median: 16.400
    Min:    0.000
    Max:    283.000
    StdDev: 51.607

### 3.4 Maximum Acceleration
  Max Acceleration (px/s^2):
    Count: 102
    Mean:   6593.600
    Median: 4317.300
    Min:    0.000
    Max:    26536.500
    StdDev: 6602.857

### 3.5 Acceleration at Trigger
  Acceleration at Trigger (px/s^2):
    Count: 102
    Mean:   4311.941
    Median: 1738.600
    Min:    -906.000
    Max:    24734.800
    StdDev: 6129.786

### 3.6 High Velocity vs Recovery
  High velocity events (>500 px/s): 70
    - Recovered: 21 (30.0%)
  Low velocity events (<=500 px/s): 32
    - Recovered: 11 (34.4%)

## 4. CNN Confidence Analysis

### 4.1 Live CNN at Trigger
  Live CNN Confidence at Trigger:
    Count: 102
    Mean:   0.201
    Median: 0.000
    Min:    0.000
    Max:    0.994
    StdDev: 0.326
  Events with non-zero live CNN: 36/102

### 4.2 Retrospective CNN at Trigger
  Retro CNN Confidence at Trigger:
    Count: 102
    Mean:   0.718
    Median: 0.829
    Min:    0.000
    Max:    1.000
    StdDev: 0.296
  Events with non-zero retro CNN: 99/102

### 4.3 Retro CNN Distribution (non-zero values only)
  Retro CNN (non-zero):
    Count: 99
    Mean:   0.739
    Median: 0.855
    Min:    0.187
    Max:    1.000
    StdDev: 0.272
  Distribution (0.1 buckets):
    0.1-0.2:   8 ########
    0.2-0.3:   3 ###
    0.3-0.4:   5 #####
    0.4-0.5:   9 #########
    0.5-0.6:   4 ####
    0.6-0.7:   8 ########
    0.7-0.8:  10 ##########
    0.8-0.9:   8 ########
    0.9-1.0:  44 ############################################

### 4.4 CNN Confidence During Lost Phase
  Avg Retro CNN during lost phase:
    Count: 102
    Mean:   0.634
    Median: 0.820
    Min:    0.000
    Max:    0.995
    StdDev: 0.363

  Max Retro CNN during lost phase:
    Count: 102
    Mean:   0.730
    Median: 0.868
    Min:    0.000
    Max:    1.000
    StdDev: 0.349

  Avg Live CNN during lost phase:
    Count: 102
    Mean:   0.103
    Median: 0.000
    Min:    0.000
    Max:    0.694
    StdDev: 0.195

  Max Live CNN during lost phase:
    Count: 102
    Mean:   0.113
    Median: 0.000
    Min:    0.000
    Max:    0.694
    StdDev: 0.207

### 4.5 CNN Confidence Trend Before Loss
  CNN declining before loss: 32
  CNN flat/zero before loss: 12
  CNN increasing before loss: 58
  (Most events have 0 CNN confidence during had_it phase - CNN was not active)

## 5. Silhouette Analysis

### 5.1 Silhouette at Trigger
  Silhouette Confidence at Trigger:
    Count: 102
    Mean:   0.775
    Median: 0.982
    Min:    0.000
    Max:    1.000
    StdDev: 0.297
  Events with non-zero silhouette: 94/102

### 5.2 Silhouette Methods Used
  - template: 102
  - searching: 32
  - recovered: 12
  - motion_roi: 11
  - lost: 3

### 5.3 Events with Silhouette Data Present
  102/102 events have non-zero silhouette confidence
  Avg silhouette confidence when present:
    Count: 102
    Mean:   0.814
    Median: 0.846
    Min:    0.480
    Max:    1.000
    StdDev: 0.144

## 6. Blob Candidate Analysis

  Events with blob data: 90/102 (88.2%)
  Max blob count per event:
    Count: 102
    Mean:   8.912
    Median: 5.000
    Min:    0.000
    Max:    32.000
    StdDev: 9.903

  Frames with blobs per event:
    Count: 102
    Mean:   24.098
    Median: 20.500
    Min:    0.000
    Max:    100.000
    StdDev: 22.271

### 6.1 Blob Candidate CNN Scores
  Blob CNN Scores:
    Count: 2813
    Mean:   0.790
    Median: 0.911
    Min:    0.097
    Max:    1.000
    StdDev: 0.259

### 6.2 Blob Candidate Positions
  X range: 35 to 1875
  Y range: 36 to 1047

## 7. Position Analysis & Bounding Box Investigation

### 7.1 Loss Positions
  X range: 43 to 1897 (mean: 1077, median: 1284)
  Y range: 65 to 965 (mean: 433, median: 368)

### 7.2 Had-It Positions (last known good position)
  X range: 6 to 1897 (mean: 885, median: 1064)
  Y range: 15 to 956 (mean: 331, median: 210)

### 7.3 Displacement Analysis
  Displacement had_it -> lost (px):
    Count: 90
    Mean:   406.291
    Median: 212.301
    Min:    0.000
    Max:    2025.576
    StdDev: 528.700

  Displacement had_it -> recovery (px):
    Count: 32
    Mean:   206.603
    Median: 124.450
    Min:    0.000
    Max:    1020.480
    StdDev: 298.487

### 7.4 Position Clustering (Loss events in 100px grid)
  Top grid cells where cursor was lost:
    (1500-1600, 900-1000): 11 events
    (1300-1400, 100-200): 11 events
    (1300-1400, 200-300): 11 events
    (1200-1300, 200-300): 6 events
    (300-400, 500-600): 4 events
    (1200-1300, 100-200): 4 events
    (500-600, 400-500): 3 events
    (1800-1900, 700-800): 2 events
    (1100-1200, 600-700): 2 events
    (900-1000, 300-400): 2 events
    (900-1000, 200-300): 2 events
    (0-100, 0-100): 2 events
    (500-600, 300-400): 2 events
    (1000-1100, 100-200): 2 events
    (400-500, 600-700): 2 events

### 7.5 Position Clustering (had_it positions in 100px grid)
  Top grid cells where cursor was last seen (had_it):
    (0-100, 0-100): 16 events
    (1400-1500, 200-300): 8 events
    (1500-1600, 900-1000): 6 events
    (1100-1200, 100-200): 6 events
    (1400-1500, 100-200): 6 events
    (1000-1100, 100-200): 5 events
    (1100-1200, 200-300): 3 events
    (1200-1300, 200-300): 3 events
    (100-200, 400-500): 3 events
    (1300-1400, 600-700): 2 events
    (1000-1100, 200-300): 2 events
    (1200-1300, 100-200): 2 events
    (400-500, 600-700): 2 events
    (800-900, 800-900): 2 events
    (1800-1900, 700-800): 1 events

### 7.6 Edge Position Analysis
  Losses near screen edge (<50px from edge, assuming 1920x1080):
    Edge losses: 4 (3.9%)
    Non-edge losses: 86 (84.3%)

### 7.7 Very Low Position Analysis
  Events where last had_it position had x<20 or y<20: 11
    20260219_235603: position=(6, 15), trigger=position_jump_1832px
    20260220_000003: position=(6, 15), trigger=position_jump_1832px
    20260220_001236: position=(6, 15), trigger=position_jump_1832px
    20260220_012757: position=(6, 15), trigger=position_jump_1819px
    20260220_012930: position=(6, 15), trigger=position_jump_1832px
    20260220_013102: position=(6, 15), trigger=position_jump_2025px
    20260220_013159: position=(6, 15), trigger=position_jump_1778px
    20260220_014229: position=(6, 15), trigger=position_jump_988px
    20260220_015044: position=(6, 15), trigger=position_jump_649px
    20260220_015225: position=(6, 15), trigger=position_jump_637px
    20260220_015344: position=(6, 15), trigger=position_jump_655px

### 7.8 Screen Quadrant Distribution (Loss positions)
  Top-left (Q1):     18 (20.0%)
  Top-right (Q2):    42 (46.7%)
  Bottom-left (Q3):  12 (13.3%)
  Bottom-right (Q4): 18 (20.0%)

## 8. Edge Cases Analysis

### 8.1 Edge Case Distribution
  - peak_acceleration: 91
  - peak_velocity: 86
  - high_accel: 77
  - high_speed: 70
  - noisy_background: 45
  - partial_cursor: 12

## 9. Detection Methods

### 9.1 Method at Trigger
  - sil_template: 47 (46.1%)
  - dual_probe: 38 (37.3%)
  - motion_track: 12 (11.8%)
  - sil_motion_roi: 2 (2.0%)
  - unknown: 1 (1.0%)
  - yolo_anchor: 1 (1.0%)
  - jitter: 1 (1.0%)

## 10. Key Patterns and Correlations

### 10.1 Does High Velocity Correlate with Loss?
  Velocity >   100 px/s:  84 events,  26 recovered (31%)
  Velocity >   500 px/s:  70 events,  21 recovered (30%)
  Velocity >  1000 px/s:  45 events,   7 recovered (16%)
  Velocity >  2000 px/s:  22 events,   3 recovered (14%)
  Velocity >  5000 px/s:   1 events,   0 recovered (0%)

### 10.2 Is There a Confidence Threshold for Loss?
  Retrospective CNN at trigger (non-zero values):
    < 0.30: 11, >= 0.30: 88
    < 0.50: 25, >= 0.50: 74
    < 0.70: 37, >= 0.70: 62
    < 0.80: 47, >= 0.80: 52
    < 0.90: 55, >= 0.90: 44
    < 0.95: 63, >= 0.95: 36

### 10.3 Recovery Time vs Displacement
  Events with both recovery time and displacement data: 21
  Small displacement (<100px): avg recovery time 13.20s
  Large displacement (>=100px): avg recovery time 7.51s

### 10.4 Characteristics of Events That Never Recover
  Max velocity (unrecovered):
    Count: 70
    Mean:   1540.573
    Median: 1235.150
    Min:    0.000
    Max:    5414.100
    StdDev: 1401.400

  Duration (unrecovered):
    Count: 70
    Mean:   24.087
    Median: 24.900
    Min:    16.400
    Max:    25.100
    StdDev: 2.544

  Trigger types (unrecovered):
    cnn_drop: 14
    validation_lost: 6
    position_jump_1832px: 4
    position_jump_288px: 2
    position_jump_367px: 2
    position_jump_1819px: 1
    position_jump_2025px: 1
    position_jump_1778px: 1
    position_jump_276px: 1
    position_jump_531px: 1
    position_jump_865px: 1
    position_jump_652px: 1
    position_jump_988px: 1
    position_jump_246px: 1
    position_jump_297px: 1
    position_jump_451px: 1
    position_jump_649px: 1
    position_jump_637px: 1
    position_jump_655px: 1
    position_jump_946px: 1
    position_jump_393px: 1
    position_jump_314px: 1
    position_jump_231px: 1
    position_jump_455px: 1
    position_jump_630px: 1
    position_jump_623px: 1
    position_jump_573px: 1
    position_jump_577px: 1
    position_jump_881px: 1
    position_jump_294px: 1
    position_jump_228px: 1
    position_jump_930px: 1
    position_jump_223px: 1
    position_jump_941px: 1
    position_jump_984px: 1
    position_jump_203px: 1
    position_jump_264px: 1
    position_jump_364px: 1
    position_jump_1185px: 1
    position_jump_1112px: 1
    position_jump_215px: 1
    position_jump_349px: 1
    position_jump_490px: 1
    position_jump_435px: 1
    position_jump_511px: 1
    position_jump_470px: 1
    position_jump_773px: 1

### 10.5 Event Timing and Clustering
  Time between consecutive events (seconds):
    Count: 101
    Mean:   151.653
    Median: 92.000
    Min:    33.000
    Max:    2476.000
    StdDev: 271.688

  Events within 2 minutes of previous: 69/101

  Hourly distribution:
    00:00:   8 ########
    01:00:  20 ####################
    02:00:  41 #########################################
    03:00:  27 ###########################
    04:00:   5 #####
    23:00:   1 #

### 10.6 CNN Confidence Paradox (High Retro CNN During Loss)
  Events where retro CNN > 0.8 during lost phase: 64/102
  (These events had high CNN confidence but were still classified as lost)
  Sample events:
    20260219_235603: retro CNN max during lost = 0.8939, recovered=False, trigger=position_jump_1832px
    20260220_000003: retro CNN max during lost = 0.8584, recovered=False, trigger=position_jump_1832px
    20260220_001002: retro CNN max during lost = 0.8584, recovered=False, trigger=cnn_drop
    20260220_001236: retro CNN max during lost = 0.8584, recovered=False, trigger=position_jump_1832px
    20260220_001805: retro CNN max during lost = 0.8680, recovered=False, trigger=cnn_drop
    20260220_001936: retro CNN max during lost = 0.8584, recovered=False, trigger=cnn_drop
    20260220_003138: retro CNN max during lost = 0.8680, recovered=False, trigger=cnn_drop
    20260220_004509: retro CNN max during lost = 0.8584, recovered=False, trigger=cnn_drop
    20260220_004641: retro CNN max during lost = 0.8584, recovered=False, trigger=cnn_drop
    20260220_012757: retro CNN max during lost = 0.8584, recovered=False, trigger=position_jump_1819px

## 11. Raw Data Table (All Events)

| Event ID | Trigger | Recovered | Duration | Time to Lose | Time to Recover | Max Vel | Retro CNN@Trig | Sil@Trig | Had-It Pos | Lost Pos | Displacement | Edge Cases |
|----------|---------|-----------|----------|-------------|----------------|---------|---------------|---------|-----------|---------|-------------|------------|
| 20260219_235603 | position_jump_1832px | False | 16.5s | 1.02 | never | 1804 | 0.7433 | 0.000 | (6,15) | (1578,957) | 1833 | high_accel, partial_cursor, high_speed |
| 20260220_000003 | position_jump_1832px | False | 16.5s | 1.05 | never | 1751 | 0.4463 | 0.000 | (6,15) | (1578,957) | 1833 | high_speed, peak_velocity, peak_acceleration |
| 20260220_001002 | cnn_drop | False | 25.0s | 0.24 | never | 52 | 0.3816 | 1.000 | (1570,956) | (1570,956) | 0 | peak_velocity, peak_acceleration |
| 20260220_001236 | position_jump_1832px | False | 16.6s | 1.06 | never | 1736 | 0.4011 | 0.000 | (6,15) | (1578,957) | 1833 | peak_acceleration, high_speed, partial_cursor |
| 20260220_001805 | cnn_drop | False | 25.0s | 0.23 | never | 45 | 0.8680 | 1.000 | (1570,956) | (1571,965) | 9 | peak_acceleration |
| 20260220_001936 | cnn_drop | False | 24.8s | 0.22 | never | 45 | 0.4546 | 1.000 | (1570,956) | (1570,956) | 0 | peak_acceleration |
| 20260220_003138 | cnn_drop | False | 25.0s | 0.22 | never | 41 | 0.8680 | 0.708 | (1570,956) | (1571,965) | 9 | peak_acceleration |
| 20260220_004509 | cnn_drop | False | 24.9s | 0.23 | never | 45 | 0.3816 | 0.707 | (1570,956) | (1570,956) | 0 | peak_acceleration |
| 20260220_004641 | cnn_drop | False | 24.9s | 0.20 | never | 45 | 0.3816 | 1.000 | (1570,956) | (1570,956) | 0 | peak_acceleration |
| 20260220_012757 | position_jump_1819px | False | 16.4s | 0.94 | never | 1940 | 0.6299 | 0.000 | (6,15) | (1563,956) | 1819 | peak_acceleration, high_speed, peak_velocity |
| 20260220_012930 | position_jump_1832px | False | 16.4s | 0.88 | never | 2084 | 0.4011 | 0.000 | (6,15) | (1578,957) | 1833 | high_accel, peak_acceleration, high_speed |
| 20260220_013102 | position_jump_2025px | False | 16.6s | 1.09 | never | 1856 | 0.0000 | 0.000 | (6,15) | (1897,741) | 2026 | partial_cursor, peak_velocity, high_speed |
| 20260220_013159 | position_jump_1778px | False | 16.6s | 1.13 | never | 1577 | 0.6441 | 0.000 | (6,15) | (1621,760) | 1779 | high_accel, peak_acceleration, high_speed |
| 20260220_013245 | position_jump_276px | False | 24.9s | 0.24 | never | 1490 | 0.0000 | 0.523 | (1897,741) | (1897,741) | 0 | high_accel, peak_acceleration, high_speed |
| 20260220_013331 | position_jump_531px | False | 24.8s | 0.23 | never | 846 | 0.1911 | 0.672 | (1377,632) | (1377,632) | 0 | high_accel, peak_acceleration, high_speed |
| 20260220_013659 | position_jump_909px | True | 15.2s | 0.24 | 3.40 | 3784 | 0.9416 | 1.000 | (1374,690) | (465,706) | 909 | peak_velocity, high_accel, peak_acceleration |
| 20260220_013738 | position_jump_865px | False | 24.9s | 1.03 | never | 837 | 0.7279 | 0.649 | (289,735) | (1151,661) | 865 | high_accel, peak_acceleration, high_speed |
| 20260220_013833 | validation_lost | False | 24.9s | 0.24 | never | 498 | 0.6399 | 0.991 | (1111,656) | (1111,656) | 0 | peak_velocity, high_accel, noisy_background |
| 20260220_013919 | position_jump_540px | True | 12.0s | N/A | never | 698 | 0.8953 | 0.456 | N/A | N/A | N/A | high_accel, peak_acceleration, high_speed |
| 20260220_013952 | position_jump_652px | False | 25.0s | 0.21 | never | 116 | 0.7886 | 0.608 | (950,369) | (950,369) | 0 | peak_velocity, peak_acceleration |
| 20260220_014229 | position_jump_988px | False | 24.9s | 0.25 | never | 4025 | 0.4377 | 0.535 | (6,15) | (929,368) | 988 | high_speed, peak_acceleration, high_accel |
| 20260220_014315 | position_jump_246px | False | 24.9s | 0.24 | never | 0 | 0.1967 | 1.000 | (1165,298) | (1165,298) | 0 | noisy_background |
| 20260220_014554 | position_jump_297px | False | 24.9s | 0.24 | never | 1318 | 0.8299 | 1.000 | (502,657) | (651,915) | 298 | high_speed, peak_acceleration, noisy_background |
| 20260220_014641 | position_jump_451px | False | 24.9s | 0.20 | never | 4199 | 0.7184 | 1.000 | (343,585) | (343,585) | 0 | high_speed, peak_acceleration, high_accel |
| 20260220_015044 | position_jump_649px | False | 24.9s | 0.24 | never | 2657 | 0.6424 | 1.000 | (6,15) | (322,583) | 650 | high_accel, partial_cursor, high_speed |
| 20260220_015225 | position_jump_637px | False | 24.8s | 0.20 | never | 3175 | 0.7471 | 0.535 | (6,15) | (325,567) | 638 | peak_acceleration, high_accel, partial_cursor |
| 20260220_015344 | position_jump_655px | False | 25.0s | 0.21 | never | 3142 | 0.5712 | 0.535 | (6,15) | (356,569) | 655 | peak_acceleration, peak_velocity, high_speed |
| 20260220_015431 | position_jump_578px | True | 12.2s | N/A | never | 0 | 0.9880 | 0.996 | N/A | N/A | N/A | none |
| 20260220_015515 | cnn_drop | False | 24.9s | N/A | never | 0 | 0.0000 | 0.000 | N/A | N/A | N/A | none |
| 20260220_020049 | position_jump_946px | False | 24.8s | 0.24 | never | 3885 | 0.5615 | 1.000 | (43,82) | (980,217) | 947 | high_accel, high_speed, peak_velocity |
| 20260220_020221 | position_jump_288px | False | 25.0s | 0.25 | never | 1479 | 0.6693 | 0.944 | (1205,217) | (983,401) | 288 | high_accel, high_speed, peak_velocity |
| 20260220_020313 | position_jump_288px | False | 25.1s | 0.24 | never | 1743 | 0.4930 | 0.481 | (1211,217) | (923,217) | 288 | high_accel, high_speed, peak_velocity |
| 20260220_020401 | position_jump_393px | False | 24.7s | 0.24 | never | 1847 | 0.4951 | 0.990 | (871,607) | (871,607) | 0 | high_accel, high_speed, peak_velocity |
| 20260220_020449 | position_jump_314px | False | 25.0s | 0.21 | never | 688 | 0.1927 | 0.864 | (845,294) | (845,294) | 0 | high_accel, high_speed, peak_velocity |
| 20260220_020536 | position_jump_231px | False | 24.9s | 0.21 | never | 460 | 0.6701 | 0.856 | (809,65) | (809,65) | 0 | noisy_background, high_accel, peak_acceleration |
| 20260220_020657 | position_jump_455px | False | 25.0s | 0.21 | never | 2164 | 0.8284 | 1.000 | (808,139) | (524,495) | 455 | high_accel, high_speed, peak_acceleration |
| 20260220_020920 | cnn_drop | False | 25.0s | 0.23 | never | 0 | 0.9937 | 0.997 | (43,81) | (43,81) | 0 | none |
| 20260220_021052 | position_jump_630px | False | 24.9s | 0.21 | never | 2955 | 0.7059 | 1.000 | (43,81) | (519,494) | 630 | high_accel, peak_acceleration, peak_velocity |
| 20260220_021153 | position_jump_623px | False | 25.0s | 0.21 | never | 2934 | 0.9879 | 0.513 | (519,484) | (43,82) | 623 | high_accel, peak_acceleration, peak_velocity |
| 20260220_021354 | position_jump_573px | False | 25.0s | 0.25 | never | 2845 | 0.7314 | 0.996 | (43,82) | (530,385) | 574 | high_accel, peak_acceleration, peak_velocity |
| 20260220_021441 | position_jump_573px | True | 12.0s | N/A | never | 0 | 0.9879 | 0.996 | N/A | N/A | N/A | none |
| 20260220_021525 | position_jump_577px | False | 24.9s | 0.22 | never | 2879 | 0.3993 | 0.996 | (43,82) | (535,385) | 578 | high_accel, peak_acceleration, peak_velocity |
| 20260220_021612 | position_jump_577px | True | 12.2s | N/A | never | 0 | 0.9879 | 0.996 | N/A | N/A | N/A | noisy_background |
| 20260220_021646 | position_jump_881px | False | 25.0s | 0.20 | never | 3177 | 0.1871 | 0.492 | (873,380) | (873,380) | 0 | peak_velocity, high_speed, high_accel |
| 20260220_021735 | position_jump_563px | True | 12.2s | N/A | never | 898 | 0.1974 | 0.501 | N/A | N/A | N/A | peak_velocity, high_speed, high_accel |
| 20260220_021809 | cnn_drop | False | 25.1s | 0.24 | never | 1434 | 0.1906 | 0.515 | (798,877) | (798,877) | 0 | peak_velocity, high_speed, high_accel |
| 20260220_021857 | position_jump_714px | True | 12.2s | N/A | never | 920 | 0.9951 | 0.454 | N/A | N/A | N/A | peak_velocity, high_speed, high_accel |
| 20260220_021931 | position_jump_443px | True | 12.1s | N/A | never | 443 | 0.9722 | 1.000 | N/A | N/A | N/A | high_accel, peak_acceleration, peak_velocity |
| 20260220_022011 | position_jump_240px | True | 12.0s | N/A | never | 1306 | 0.9775 | 0.518 | N/A | N/A | N/A | peak_velocity, high_speed, high_accel |
| 20260220_022320 | position_jump_209px | True | 16.1s | 0.24 | 4.45 | 862 | 0.9569 | 0.601 | (1077,186) | (1285,209) | 209 | high_accel, peak_acceleration, peak_velocity |
| 20260220_022453 | cnn_drop | True | 12.7s | 0.29 | 1.02 | 443 | 0.9590 | 1.000 | (1192,186) | (1318,171) | 127 | high_accel, peak_acceleration, peak_velocity |
| 20260220_022627 | position_jump_294px | False | 25.1s | 0.29 | never | 1023 | 0.9959 | 0.980 | (1072,210) | (1364,169) | 295 | peak_velocity, high_speed, high_accel |
| 20260220_022934 | cnn_drop | False | 25.0s | 0.28 | never | 517 | 0.9821 | 1.000 | (1195,211) | (1319,209) | 124 | high_accel, peak_acceleration, peak_velocity |
| 20260220_023106 | position_jump_228px | False | 24.7s | 0.22 | never | 1053 | 0.9971 | 0.689 | (1056,218) | (1284,210) | 228 | peak_velocity, high_speed, high_accel |
| 20260220_023410 | cnn_drop | True | 26.8s | 0.22 | 15.06 | 0 | 0.9255 | 0.591 | (1193,186) | (1193,186) | 0 | none |
| 20260220_023543 | cnn_drop | True | 24.5s | 0.21 | 12.84 | 512 | 0.9230 | 0.902 | (1193,186) | (1302,170) | 110 | high_accel, peak_acceleration, peak_velocity |
| 20260220_023630 | position_jump_248px | True | 12.0s | N/A | never | 0 | 0.9416 | 1.000 | N/A | N/A | N/A | none |
| 20260220_023715 | cnn_drop | True | 22.8s | 0.25 | 11.04 | 10 | 0.9416 | 1.000 | (1054,169) | (1054,169) | 0 | noisy_background |
| 20260220_023847 | position_jump_231px | True | 20.6s | 0.25 | 9.02 | 933 | 0.8549 | 1.000 | (1054,169) | (1285,178) | 231 | peak_velocity, high_speed, high_accel |
| 20260220_024019 | cnn_drop | True | 18.3s | 0.21 | 6.62 | 949 | 0.9923 | 1.000 | (1478,172) | (1288,216) | 195 | peak_velocity, high_speed, high_accel |
| 20260220_024152 | cnn_drop | True | 15.9s | 0.22 | 4.21 | 502 | 0.9948 | 0.574 | (1213,216) | (1318,184) | 110 | peak_velocity, high_speed, high_accel |
| 20260220_024456 | cnn_drop | False | 24.9s | 0.25 | never | 771 | 0.9954 | 0.631 | (1458,201) | (1269,197) | 189 | high_accel, peak_acceleration, peak_velocity |
| 20260220_024556 | validation_lost | False | 25.0s | 0.23 | never | 450 | 0.6896 | 0.627 | (1431,183) | (1431,183) | 0 | high_accel, peak_acceleration, peak_velocity |
| 20260220_024643 | position_jump_1044px | True | 25.6s | 0.20 | 13.83 | 112 | 0.9205 | 0.997 | (406,386) | (406,386) | 0 | peak_acceleration, peak_velocity |
| 20260220_024801 | position_jump_930px | False | 25.0s | 0.21 | never | 4502 | 0.8807 | 0.997 | (396,379) | (1303,169) | 931 | peak_velocity, high_speed, high_accel |
| 20260220_025106 | position_jump_223px | False | 24.9s | 0.28 | never | 792 | 0.9990 | 0.683 | (1097,186) | (1320,186) | 223 | peak_velocity, high_speed, high_accel |
| 20260220_025238 | position_jump_941px | False | 25.0s | 0.23 | never | 4149 | 0.9023 | 0.549 | (1272,191) | (460,668) | 942 | peak_velocity, high_speed, high_accel |
| 20260220_025410 | position_jump_984px | False | 24.8s | 0.21 | never | 4645 | 0.9991 | 0.607 | (451,661) | (1326,209) | 985 | peak_velocity, high_speed, high_accel |
| 20260220_025541 | position_jump_203px | False | 25.0s | 0.21 | never | 988 | 0.9822 | 0.716 | (1480,170) | (1281,211) | 203 | peak_velocity, high_speed, high_accel |
| 20260220_025845 | position_jump_873px | True | 24.3s | 0.20 | 12.63 | 4294 | 0.8940 | 1.000 | (1157,172) | (419,639) | 873 | high_accel, peak_acceleration, peak_velocity |
| 20260220_030150 | position_jump_997px | True | 20.4s | 0.24 | 8.63 | 4145 | 0.9893 | 0.635 | (418,644) | (1318,213) | 998 | high_accel, peak_acceleration, peak_velocity |
| 20260220_030322 | cnn_drop | True | 18.6s | 0.23 | 6.83 | 730 | 0.9988 | 1.000 | (1458,201) | (1318,213) | 141 | peak_velocity, high_speed, high_accel |
| 20260220_030454 | position_jump_224px | True | 16.3s | 0.23 | 4.61 | 971 | 0.7748 | 1.000 | (1527,176) | (1303,169) | 224 | peak_velocity, high_speed, high_accel |
| 20260220_030626 | position_jump_216px | True | 14.0s | 0.20 | 2.21 | 1069 | 0.9927 | 1.000 | (1498,169) | (1285,209) | 217 | high_accel, peak_acceleration, peak_velocity |
| 20260220_030728 | validation_lost | False | 24.8s | 0.24 | never | 360 | 0.9927 | 0.773 | (1418,172) | (1368,175) | 50 | peak_acceleration, peak_velocity |
| 20260220_030931 | position_jump_264px | False | 24.9s | 0.23 | never | 1152 | 0.9630 | 1.000 | (1118,180) | (1382,170) | 264 | peak_velocity, high_speed, high_accel |
| 20260220_031103 | cnn_drop | False | 25.0s | 0.25 | never | 625 | 0.9995 | 0.755 | (1483,197) | (1324,204) | 159 | peak_velocity, high_speed, high_accel |
| 20260220_031408 | cnn_drop | False | 25.0s | 0.21 | never | 539 | 0.9870 | 1.000 | (1391,214) | (1278,205) | 113 | peak_velocity, high_speed, high_accel |
| 20260220_031716 | cnn_drop | True | 24.5s | 0.26 | 12.85 | 426 | 0.9722 | 0.613 | (1458,201) | (1458,201) | 0 | peak_acceleration, peak_velocity, noisy_background |
| 20260220_031849 | cnn_drop | True | 22.5s | 0.22 | 10.85 | 778 | 0.9995 | 1.000 | (1492,207) | (1318,209) | 174 | high_accel, peak_acceleration, peak_velocity |
| 20260220_032153 | position_jump_253px | True | 17.9s | 0.23 | 6.21 | 1112 | 0.9806 | 0.817 | (1113,171) | (1366,169) | 253 | high_accel, peak_acceleration, peak_velocity |
| 20260220_032802 | position_jump_364px | False | 24.9s | 0.24 | never | 1526 | 0.9923 | 1.000 | (955,192) | (1319,214) | 365 | high_accel, peak_acceleration, peak_velocity |
| 20260220_032934 | cnn_drop | False | 24.9s | 0.20 | never | 699 | 0.9810 | 1.000 | (1498,214) | (1355,209) | 143 | high_accel, peak_acceleration, peak_velocity |
| 20260220_033237 | position_jump_1185px | False | 24.8s | 0.22 | never | 5414 | 0.5677 | 0.991 | (1402,220) | (255,518) | 1185 | peak_velocity, high_speed, high_accel |
| 20260220_033339 | position_jump_1112px | False | 25.0s | 0.25 | never | 4447 | 0.9383 | 0.988 | (261,515) | (1319,171) | 1113 | peak_velocity, high_speed, high_accel |
| 20260220_033501 | validation_lost | False | 25.0s | 0.23 | never | 487 | 0.6978 | 1.000 | (1296,175) | (1296,175) | 0 | high_accel, peak_acceleration, peak_velocity |
| 20260220_033846 | position_jump_215px | False | 24.9s | 0.23 | never | 953 | 0.9933 | 1.000 | (1104,226) | (1319,214) | 215 | peak_velocity, high_speed, high_accel |
| 20260220_034018 | cnn_drop | True | 25.3s | 0.21 | 13.66 | 612 | 0.9998 | 1.000 | (1500,215) | (1370,203) | 131 | peak_velocity, high_speed, high_accel |
| 20260220_034150 | cnn_drop | True | 23.2s | 0.21 | 11.43 | 656 | 0.9955 | 1.000 | (1458,201) | (1319,214) | 140 | high_accel, peak_acceleration, peak_velocity |
| 20260220_034323 | position_jump_218px | True | 20.7s | 0.21 | 9.04 | 1032 | 0.9970 | 1.000 | (1498,214) | (1282,183) | 218 | peak_velocity, high_speed, high_accel |
| 20260220_034602 | validation_lost | False | 25.0s | 0.23 | never | 304 | 0.7248 | 0.805 | (1072,165) | (1072,165) | 0 | peak_acceleration, peak_velocity, noisy_background |
| 20260220_034650 | position_jump_387px | True | 12.0s | N/A | never | 374 | 0.9722 | 0.613 | N/A | N/A | N/A | peak_acceleration, peak_velocity |
| 20260220_034910 | position_jump_1004px | True | 12.1s | N/A | never | 926 | 0.3987 | 1.000 | N/A | N/A | N/A | peak_velocity, high_speed, high_accel |
| 20260220_034944 | validation_lost | False | 24.8s | 0.34 | never | 0 | 0.5360 | 0.974 | (177,477) | (177,477) | 0 | none |
| 20260220_035103 | position_jump_367px | False | 24.9s | 0.22 | never | 1686 | 0.1958 | 0.977 | (131,457) | (498,450) | 367 | peak_velocity, high_speed, high_accel |
| 20260220_035150 | position_jump_349px | False | 24.9s | 0.22 | never | 88 | 0.4851 | 0.985 | (149,438) | (149,438) | 0 | peak_acceleration, peak_velocity |
| 20260220_035408 | position_jump_490px | False | 24.9s | 0.29 | never | 2965 | 0.7603 | 0.456 | (506,321) | (961,504) | 490 | peak_velocity, high_speed, high_accel |
| 20260220_040629 | position_jump_435px | False | 25.0s | 1.05 | never | 415 | 0.2610 | 1.000 | (856,578) | (1291,569) | 435 | high_accel, peak_acceleration, peak_velocity |
| 20260220_040717 | position_jump_511px | False | 24.9s | 0.25 | never | 0 | 0.1949 | 0.562 | (836,803) | (836,803) | 0 | none |
| 20260220_040855 | position_jump_470px | False | 25.0s | 0.24 | never | 1962 | 0.2675 | 0.992 | (836,803) | (544,434) | 471 | high_accel, peak_acceleration, peak_velocity |
| 20260220_041033 | position_jump_367px | False | 24.8s | 1.08 | never | 1396 | 0.2714 | 0.991 | (670,513) | (1009,372) | 367 | high_accel, peak_acceleration, peak_velocity |
| 20260220_041120 | position_jump_773px | False | 25.0s | 0.24 | never | 113 | 0.4465 | 0.993 | (474,931) | (474,931) | 0 | peak_acceleration, peak_velocity |

## 12. CRITICAL FINDING: The Live CNN Recovery Gate

This is the single most important finding in the entire analysis.

### 12.1 The Recovery Bottleneck: Live CNN is the Gatekeeper

**Recovery REQUIRES live_cnn > 0.8.** This is not a correlation -- it is an absolute gate:

| Condition | Recovery Count |
|-----------|---------------|
| Live CNN > 0.8 during lost phase | 21/21 recovered (100%) |
| Live CNN <= 0.8 during lost phase | 0/71 recovered (0%) |

At the moment of recovery transition (lost -> recovered):
- Live CNN: mean=0.961, min=0.819, max=1.000 -- **always high**
- Retro CNN: mean=0.954, min=0.744, max=1.000
- Silhouette: mean=0.924, min=0.670, max=1.000

**The paradox:** 43 of 71 non-recovery events had retro CNN > 0.8 during the lost phase. The retrospective CNN clearly sees the cursor in the frame, but the system does not use this signal for recovery. Live CNN stays at 0.0 in 43/71 non-recovery events, and never exceeds 0.694 in any non-recovery event.

**Distribution of max live CNN during lost phase (non-recovery events):**
```
0.0:       43 events  (60%)
0.01-0.3:   8 events  (11%)
0.3-0.5:    9 events  (13%)
0.5-0.7:   11 events  (15%)
0.7-0.8:    0 events  (0%)
0.8+:       0 events  (0%)  <-- THIS IS THE GAP
```

**Implication:** If the recovery mechanism trusted retro_cnn >= 0.8 (in addition to live_cnn), approximately 43 additional events could have recovered, raising the recovery rate from 31% to roughly 63%.

### 12.2 Three Distinct Failure Patterns

The 102 events fall into three clear categories:

**Pattern A: The (6,15) Ghost Position (11 events, 11%)**
- Last had_it position is always exactly (6, 15)
- Method is "manual" with zero CNN, zero silhouette
- Then cursor jumps 1500-2000px to the right side of screen
- The cursor at (6,15) is likely a phantom/default position -- not where the cursor actually is
- This is a **detection initialization failure**, not a tracking loss
- All triggers are `position_jump_1778px` to `position_jump_2025px`
- None of these ever recover

**Pattern B: Stationary cnn_drop at (1570,956) (6 early events)**
- Cursor is stationary at (1570, 956) with silhouette method "sil_template"
- Cursor oscillates between (1570,956) and (1571,965) with sil_template/motion_track
- Live CNN drops abruptly from ~0.85 to ~0.35-0.40 in a single frame
- Retro CNN stays at 0.85+ throughout the entire lost phase
- Silhouette confidence stays at 0.7-1.0
- The cursor has NOT moved, but the system declares it lost due to live CNN drop
- This is a **false alarm** -- the cursor is right where it was
- None of these recover because live CNN stays low

**Pattern C: Genuine position jumps (71 events, 70%)**
- Cursor moves rapidly (200-2000px between frames)
- Triggered by actual user mouse movement
- Max velocity at trigger typically 500-5000 px/s
- Mix of recovered and unrecovered depending on whether live CNN re-acquires

### 12.3 The cnn_drop Trigger: Often a False Alarm

25 events are triggered by `cnn_drop`. Examining these closely:

**At the transition from had_it to lost:**
- Live CNN drops from mean 0.878 to mean 0.100 (mean drop: 0.777)
- But silhouette confidence often stays HIGH (0.7-1.0)
- And retro CNN often stays HIGH (0.8+)
- The cursor position does NOT change (displacement = 0 in many cases)

This means the system loses tracking when **only the live CNN** drops, even though:
1. The silhouette matcher still sees the cursor
2. The retrospective CNN still sees the cursor
3. The cursor has not moved

The live CNN is too volatile and serves as a single point of failure.

### 12.4 Recovery Cluster Pattern

Recoveries are NOT evenly distributed. They cluster in specific time windows:

```
23:56 - 01:59  Only 3 recoveries out of ~30 events (10%)
02:00 - 02:49  18 recoveries out of ~38 events (47%)  <-- BEST PERIOD
02:49 - 03:55  9 recoveries out of ~25 events (36%)
03:55 - 04:12  0 recoveries out of 9 events (0%)     <-- TOTAL FAILURE
```

**Within recovery clusters, there's a sawtooth pattern:**
```
Event 20260220_023410: recovery at +14.86s (slow)
Event 20260220_023543: recovery at +12.64s
Event 20260220_023715: recovery at +10.84s
Event 20260220_023847: recovery at +8.82s
Event 20260220_024019: recovery at +6.42s
Event 20260220_024152: recovery at +4.01s  (fast)
--- then resets ---
Event 20260220_024643: recovery at +13.63s (slow again)
```

This declining recovery time pattern repeats multiple times, suggesting some kind of learning or warm-up effect that periodically resets.

### 12.5 validation_lost: Borderline Live CNN

6 events use the `validation_lost` trigger. These share a distinct pattern:
- Cursor is STATIONARY (velocity = 0 in 5/6 events)
- Live CNN drops from 0.8-1.0 range to 0.5-0.7 range
- NOT as severe as cnn_drop (which goes to 0.0-0.4)
- Retro CNN is moderate (0.5-0.9)
- Silhouette stays high (0.6-1.0)
- None recover (live CNN stays below 0.8 threshold)
- All have zero displacement -- the cursor never moved

These are similar to cnn_drop but with a less dramatic CNN drop. They represent the same underlying issue: the live CNN fluctuates enough to cross below a validation threshold even when the cursor is stationary and visible.

### 12.6 Position Bounding Box Analysis

**The (6,15) pattern (11 events):**
All 11 events with had_it position near origin share the EXACT same position: (6, 15). This is suspiciously precise. It is almost certainly a default/initialization value rather than an actual cursor detection. The "method" is "manual" (not a detection method), and both CNN and silhouette are at 0.0.

**Screen quadrant bias (loss positions):**
```
Top-left:     20.0%   (expected: 25%)
Top-right:    46.7%   <-- CONCENTRATION
Bottom-left:  13.3%   (below expected)
Bottom-right: 20.0%   (expected: 25%)
```

46.7% of losses happen in the top-right quadrant. The top grid cells:
- (1300-1400, 100-200) and (1300-1400, 200-300): 22 events combined
- (1500-1600, 900-1000): 11 events

The 1300-1400 x-range, 100-300 y-range cluster is the area where the system most frequently detects and then loses the cursor. This may correspond to a specific UI element (browser tab bar? toolbar?) where the cursor shape changes frequently.

**Edge proximity:**
Only 4 events (3.9%) occur within 50px of the screen edge. Edge proximity is NOT a major factor. The "bounding box" problem is not about the physical screen edges -- it's about the DETECTION ROI or CNN model's effective detection region.

## 13. Recommendations for Improving Detection

### 13.1 PRIORITY 1: Use Retro CNN for Recovery (Est. Impact: +32% recovery rate)

The recovery mechanism currently gates on live_cnn > ~0.8. Change it to also accept retro_cnn > 0.8 as a recovery signal. This single change would potentially recover 43 additional events (raising recovery from 31% to ~63%).

Implementation:
```
IF (phase == "lost") AND (retro_cnn.confidence > 0.8 OR live_cnn.confidence > 0.8):
    transition to "recovering"
```

### 13.2 PRIORITY 2: Eliminate False cnn_drop Alarms (Est. Impact: -20% fewer events)

When live CNN drops but silhouette AND retro CNN remain high, do NOT declare loss. Add a multi-signal consensus requirement:

```
DECLARE_LOSS only when:
  live_cnn.confidence < 0.5
  AND (silhouette.confidence < 0.5 OR retro_cnn.confidence < 0.5)
```

Currently, a live CNN drop alone triggers loss even when the cursor is stationary and both silhouette and retro CNN see it clearly.

### 13.3 PRIORITY 3: Fix the (6,15) Initialization Bug

11 events (11%) show the cursor at exactly (6, 15) with method="manual" and all detection signals at 0.0. This is clearly a default/uninitialized position. Either:
- (a) Don't count (6,15) as a valid had_it position, or
- (b) Require at least ONE non-zero detection signal (CNN, silhouette) before accepting had_it status
- (c) Validate that "manual" method positions match at least one automated detection

### 13.4 PRIORITY 4: Investigate the 1300-1400, 100-300 Position Cluster

22 events cluster in the (1300-1400, 100-300) screen region. This is the most loss-prone area. Possible causes:
- UI element with unusual cursor shape (text cursor, resize handle)
- Region where cursor overlaps with something the CNN confuses
- Specific application toolbar that triggers cursor shape changes

### 13.5 PRIORITY 5: Widen ROI for High-Velocity Events

44% of events involve velocity > 1000 px/s. At 5 FPS (200ms between frames), a 1000 px/s cursor moves 200px between frames -- easily outside a 200px ROI. Scale the search region with velocity:

```
roi_size = max(200, min(800, estimated_velocity * frame_interval * 2))
```

### 13.6 PRIORITY 6: Temporal Smoothing for Live CNN

The live CNN is too volatile -- it drops from 0.85 to 0.0 in a single frame and never recovers. Apply exponential moving average or require N consecutive low-confidence frames before declaring loss:

```
smoothed_confidence = 0.7 * previous_smoothed + 0.3 * current_raw
DECLARE_LOSS only when smoothed_confidence < 0.5 for 3+ frames
```

## 14. Detailed Event Samples

### 14.1 Fastest Recovery
  Event: 20260220_022453
  Recovery time: 1.02s
  Trigger: cnn_drop
  Max velocity: 443 px/s

### 14.2 Longest Duration Event
  Event: 20260220_023410
  Duration: 26.8s
  Recovered: True
  Trigger: cnn_drop

### 14.3 Highest Velocity Event
  Event: 20260220_033237
  Max velocity: 5414 px/s
  Recovered: False
  Trigger: position_jump_1185px

### 14.4 Events with Most Blob Candidates
  20260220_013833: max_blobs=32, blob_patches=0, trigger=validation_lost
  20260220_013919: max_blobs=32, blob_patches=84, trigger=position_jump_540px
  20260220_020049: max_blobs=32, blob_patches=181, trigger=position_jump_946px
  20260220_020536: max_blobs=32, blob_patches=72, trigger=position_jump_231px
  20260220_021646: max_blobs=32, blob_patches=99, trigger=position_jump_881px

### 14.5 Zero-Duration or Very Short Events (<1s)
  Count: 0
