# Source guide

## From a recording to a report

The React client sends assessment requests to the local FastAPI service. The service and command-line entry point both call the Python analysis pipeline. The pipeline reads video timestamps, tracks people, estimates angles and supplies measurements and task conditions to the scoring modules. The reporting layer writes the results that the dashboard opens for review.

| Start here | Responsibility |
| --- | --- |
| [Web client](web/src/) and [HTTP service](api/main.py) | Create and reopen assessments, manage study folders, schedule analysis, return recorded evidence and export reports. |
| [Command line](main.py) and [batch processing](batch_process.py) | Run analyses outside the dashboard. Use `main.py process --help` for documented arguments. |
| [Video processing](pipeline/video_processor.py) and [video input](pipeline/video_input.py) | Decode frames, run detection, assemble per-person observations and orchestrate scoring and output. |
| [Landmark processing](core/) | Track people, compute angles and posture conditions, and choose evidence from synchronized cameras. |
| [Method inputs](ergonomics/adapters.py) and [scoring](ergonomics/) | Translate observations into method-specific inputs. RULA, REBA and RNLE calculators can also be called directly with supplied inputs. |
| [Task dynamics](temporal/task_dynamics.py) and [temporal summaries](temporal/aggregation.py) | Estimate motion-dependent conditions and summarize scores over time. |
| [Output writers](reporting/writers.py) and [PDF builder](reporting/document_builder.py) | Save frame evidence, trial summaries and reports. |
| [Local assistant](assistant/) | Discover models, pass authorized frames to the configured backend, retrieve method documentation and save interpretations for review. |
| [Dashboard utilities](dashboard/) | Read study results and provide local launchers, visualization and the Streamlit interface. |
| [Tests](tests/) and [research validation tools](validation/) | Exercise calculator boundaries, integration and user workflows, or prepare independent human comparisons using your own study records. |

## Files written by an analysis

| File | Contents |
| --- | --- |
| `run_config.json` | Configuration recorded for the run. Preserve it with the results when reproducing an assessment. |
| `frame_results.jsonl` | Detailed observations by frame, including anatomical-side results, intermediate calculations and validity information. |
| `frame_results.csv` | A compact index for tabular analysis or locating a frame in the detailed results. |
| `video_summary.json` | Timing, method completion, summary metrics, inputs and provenance for the trial. |
| `ASSESSMENT_REPORT.md` | Analysis report written by the processing pipeline. The dashboard can also render a PDF from saved results. |

Keep study recordings, the project’s `.posture/workflow/` directory and `results/` together when backing up an installation. Keep participant records, local settings, generated reports and downloaded models outside version control. The two examples deliberately included in the repository use synthetic inputs. [Installation and data locations](INSTALL.md)

## Reproduce the release checks

Create the Python environment, install development dependencies and build the web client as described in the installation guide. Run the following from the project root with PowerShell.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$env:ERGOPARTNER_TEST_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
node web/scripts/check-assessment-library.mjs
node web/scripts/check-camera-setup.mjs
node web/scripts/check-focused-review.mjs
```

The bundled video exercises decoding and missing-pose handling. Some regression and assistant-evidence tests require an existing local baseline or an analyzed trial. They skip when those inputs are absent. Physical camera performance and agreement with experts should be evaluated for the intended setup.
