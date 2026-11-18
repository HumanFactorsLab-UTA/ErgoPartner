# Scoring methods and measurement limits

These notes describe the code shipped in version 5.1.0. RULA and REBA screen observed working postures. The Revised NIOSH Lifting Equation (RNLE) evaluates the demands of applicable lifting tasks. A posture score alone cannot establish whether a particular worker will suffer an injury.

## From landmarks to inputs

1. **Sample and track.** The video pipeline decodes source timestamps and samples at the configured rate, 10 fps by default. The pose model provides 33 body landmarks per detected person. A separate hand model can supply 21 landmarks per detected hand. Hands, wrists or body segments can be unavailable even when a person is detected. [Video pipeline](pipeline/video_processor.py) · [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
2. **Estimate angles.** Joint and segment vectors supply measurements for the arms, wrists, legs, neck and trunk. The quality policy can admit a pose-only wrist estimate if hand landmarks are missing. The default neck-flexion proxy includes lateral inclination. The configured neutral tolerance can move a small angle into the neutral scoring band. The default posture-condition baseline is estimated within the trial, so a sustained non-neutral posture can influence the reference. [Angle calculations](core/angle_calculator.py) · [Posture conditions](core/posture_conditions.py) · [Configuration](config.yaml)
3. **Handle camera views.** Single-view depth is model-estimated. In multi-camera analysis the pipeline checks decoder timestamps (default tolerance 0.050 seconds) and resolves angles using camera roles, visibility and valid observations. This workflow does not perform calibrated geometric triangulation. Use synchronized recordings of one identifiable person per view. [Camera fusion](core/multi_view_fusion.py)
4. **Resolve task conditions.** The posture calculations use the first available source in this order: entered value, measured condition, configured profile, unknown. The default light-work profile assumes zero handled load and good coupling. Use observed task inputs or set `task_inputs.assumption_profile` to `null` to disable profile assumptions. RNLE has its own inputs and applicability review. [Resolver](ergonomics/condition_resolution.py) · [Profiles](ergonomics/profiles.py)
5. **Score and record provenance.** Each anatomical side receives a method result including raw inputs, intermediate table values, validity information and any missing inputs. A bilateral worst-side result requires numeric scores for both sides. A tie is labeled `both`. [Input adapters](ergonomics/adapters.py) · [Bilateral selection](ergonomics/common.py)

## RULA

**Group A** uses upper arm, lower arm, wrist posture and wrist twist. **Group B** uses neck, trunk and legs. Posture adjustments account for conditions such as shoulder elevation or neck rotation. Table A and Table B supply posture scores. In this implementation, the same muscle-use and force/load adjustments are added to each group. Table C combines the adjusted values into a grand score. The implementation uses one set of these adjustments per side and has no separate input for seated trunk support.

| Grand score | Interpretation used in the application |
| --- | --- |
| 1–2 | Acceptable if the posture is not maintained or repeated for long periods |
| 3–4 | Investigate, changes may be required |
| 5–6 | Investigate and change soon |
| 7 | Investigate and change immediately |

The neutral worked test supplies explicit joint positions, supported legs, intermittent zero load and no muscle-use adjustment. Table A and Table B each return 1. Table C returns **1**. [RULA calculator](ergonomics/rula.py) · [Executable case](tests/golden/test_method_golden_cases.py) · [McAtamney and Corlett, 1993](https://doi.org/10.1016/0003-6870(93)90080-S)

## REBA

**Group A** combines trunk, neck and legs. Add the load/force adjustment. **Group B** combines upper arm, lower arm and wrist. Add the coupling adjustment for the hand-to-object grip. Table C combines both adjusted groups. Activity can add points for static posture, repetition, and rapid large posture changes or an unstable base. The final bands are 1 negligible, 2–3 low, 4–7 medium, 8–10 high and 11–15 very high. The worked test gives **8** for adjusted A, **5** for adjusted B, **10** from Table C and **11** after the activity adjustment. [REBA calculator](ergonomics/reba.py) · [Executable case](tests/golden/test_method_golden_cases.py) · [Hignett and McAtamney, 2000](https://doi.org/10.1016/S0003-6870(99)00039-3)

**Check asymmetric poses and activity flags manually.** The video adapter pairs each side’s knee angle with that side’s arm. A bent left knee and a raised right arm can produce a different result from a worksheet that uses one shared leg posture with the higher-risk arm. The static-posture estimator measures elapsed time between available observations, so a tracking gap can make a hold look longer than the evidence supports. Repetitive movement such as walking can also influence the repetition estimate. Review the original frames and override a condition when your observation warrants it. [Bilateral inputs](ergonomics/adapters.py) · [Task dynamics](temporal/task_dynamics.py)

## Revised NIOSH Lifting Equation

The video pipeline calculates an origin assessment for applicable single-task lifting. The library also provides a two-location calculation for tasks requiring significant control at the destination, in which the lower recommended weight limit controls. Use the applicability review to identify unsupported conditions such as one-handed lifting, carrying or pulling, or multiple-task jobs. This release does not compute a Composite Lifting Index. [NIOSH manual](https://doi.org/10.26616/NIOSHPUB94110revised092021) · [Calculator and applicability rules](ergonomics/niosh.py)

**Recommended Weight Limit (RWL)** = 23 kg × HM × VM × DM × AM × FM × CM. **Lifting Index (LI)** = handled load in kg divided by RWL in kg.

| Multiplier | Input and calculation in the metric implementation |
| --- | --- |
| HM | Horizontal hand distance H in cm. Uses 25/H with the method’s 25 cm lower floor. |
| VM | Vertical hand height V in cm. Uses 1 − 0.003 × the absolute difference between V and 75. |
| DM | Vertical hand travel D in cm. Uses 0.82 + 4.5/D with the method’s 25 cm lower floor. |
| AM | Asymmetry angle A in degrees. Uses 1 − 0.0032 × A. |
| FM | Table value based on lifts per minute, work-duration class and hand height. |
| CM | Table value based on grip quality and hand height. |

For a reviewed applicable lift at H = 25 cm, V = 75 cm, D = 25 cm, A = 0°, 0.2 lifts/min, short duration and good coupling, all multipliers equal 1. **RWL = 23 kg**. With a 23 kg load, **LI = 1.0**. An LI above 1 means the handled load exceeds the calculated RWL for those inputs. [Worked RNLE example](tests/golden/test_method_golden_cases.py)

Entered measurements take priority. When enabled and the required observations are available, the estimator can use detected lifts and recorded participant stature to fill missing geometry or frequency. Distances filled this way are labeled `anthropometric_estimate`. Handled mass cannot be inferred from video. Job duration describes exposure beyond the clip and may be assumed when absent. Review the run configuration, assumptions and applicability fields before citing a result. [Input estimation](ergonomics/rnle_estimation.py) · [Pipeline calculation](pipeline/video_processor.py)

## Summaries and coverage

- **Maximum and 90th percentile** use numeric scored observations. The maximum includes a source frame and timestamp. The percentile is configurable.
- **Duration-weighted mean** weights a score by the elapsed time to the next scored observation of the same person. Intervals exceeding the configured maximum gap are excluded.
- **Sustained periods** require observations at or above the threshold, at least the configured duration apart from first to last, with no intervening gap over the configured maximum. Defaults are RULA ≥5 or REBA ≥8, 1.0 second minimum duration, and a 0.25-second maximum gap.
- **Method completion rate** is scored method observations divided by total method observations recorded by the completion tracker. The report also records common blocking inputs. A brief high-scoring interval and a clip with very low coverage warrant different interpretations.

See the [summary implementation](temporal/aggregation.py), [completion tracker](reporting/diagnostics.py) and [configuration](config.yaml). Worked tests check software behavior under specified inputs. Quantifying agreement with ergonomists or calibrated motion capture requires independent reference observations for the intended task.
