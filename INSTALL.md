# Install ErgoPartner on Windows

Download the version specified in your written approval and extract the complete archive to a writable folder. The web dashboard, analysis pipeline and reports run locally.

## System requirements

| Component | Requirement or tested configuration |
| --- | --- |
| Operating system | 64-bit Windows. Release checks ran on Windows 11 Enterprise build 26200. |
| Python | CPython **3.12.x x64**, including the Windows launcher. Tested with 3.12.10. |
| Web tools | Node.js **22 or newer** with npm. Tested with Node 24.22.2 and npm 11.19.1. |
| Video and PDF | FFmpeg on PATH for browser playback. Tested with FFmpeg 8.0. Setup installs Playwright's matching Chromium for PDF export. |
| Core analysis | Runs on CPU. The test workstation had a 32-core Threadripper PRO 3975WX and 256 GiB RAM. A lower CPU and RAM limit for sustained video analysis has yet to be established. |
| Storage | The tested development environment and web dependencies occupied about 1.7 GiB. Allow extra disk space for installation, the downloaded models, recordings and reports. |
| Internet | Initial setup downloads Python packages, web dependencies and a browser. The first analysis downloads the configured MediaPipe models and checks their SHA-256 hashes. |

Install Python, Node.js and FFmpeg from their publishers, then open a new PowerShell window in the folder containing `config.yaml`. [Python for Windows](https://www.python.org/downloads/windows/) · [Node.js](https://nodejs.org/en/download) · [FFmpeg](https://ffmpeg.org/download.html#build-windows)

### Install and launch

```powershell
py -3.12 -V
node --version
npm.cmd --version
ffmpeg -version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade "pip>=26.2"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
npm.cmd ci --prefix web --include=dev
npm.cmd run build --prefix web
New-Item -ItemType Directory -Force ..\studies | Out-Null
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\dashboard\launch_dashboard.ps1
```

The launcher opens Home at `http://127.0.0.1:8000/`. Running the following check from PowerShell should return `True`.

```powershell
(Invoke-RestMethod http://127.0.0.1:8000/api/health).ok
```

Use the launch command on subsequent visits. After saving your work and finishing capture or analysis, stop the local dashboard with its companion script.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\dashboard\stop_dashboard.ps1
```

### First assessment

1. Open **Analyze recorded video** or **Record live**. Create a study, enter a participant identifier and name the task. **Assessments → Add task** adds another activity under the same participant.
2. Add a single recording, or up to three synchronized views of the same person and task. Supply the task inputs you observed, including load and coupling where applicable.
3. Run analysis. Open **Summary** and **Posture review** to inspect the contributing frames, assumptions and missing inputs.
4. Use **Findings & report** to save your review and export a PDF. The bundled synthetic video contains no person and can verify installation. To evaluate posture scoring, use an authorized recording with a person visible.

The quick start uses the sibling `studies` folder. To select another library, open **Settings → Studies folder → Browse folder → Use this folder**. Keep the library, the project's `.posture/workflow/` directory and `results/` together in backups. Install updates into a fresh folder and preserve those records when removing a previous installation.

### Optional local AI with Ollama

Install Ollama for Windows and open a new PowerShell window. The default app endpoint is `http://127.0.0.1:11434/v1`. The configured model is `gemma4:26b`. A smaller local vision option is `gemma4:e4b`. [Ollama for Windows](https://ollama.com/download/windows) · [Ollama system requirements](https://docs.ollama.com/windows)

```powershell
ollama --version
ollama pull gemma4:e4b
ollama list
(Invoke-RestMethod http://127.0.0.1:11434/api/tags).models.name
```

In **Settings → Local vision model**, choose **Refresh models**, select the installed tag, and choose **Save defaults**. Open an assessment with eligible frames and request a visual interpretation. Compare its observations with the footage. Assessment chat also needs model tool support.

On the release workstation, the installed model files occupied about 8.9 GiB for `gemma4:e4b` and 16.8 GiB for `gemma4:26b`. These are disk sizes. Available memory and context size affect inference speed and whether the model can load. An external endpoint receives the frames and assessment information sent to it. Use an endpoint authorized for your study.

### Troubleshooting

| Symptom | Check |
| --- | --- |
| Python 3.12 cannot be found | Run `py -0p`. Install 64-bit CPython 3.12 with its launcher, reopen PowerShell and repeat setup. |
| Video will not replay | Check `ffmpeg -version` in the same shell. Add the directory containing ffmpeg.exe to PATH and restart the app. |
| PDF export cannot find Chromium | From the project root run `node web/scripts/install-browsers.mjs` after installing web dependencies. |
| Port 8000 is busy | Save active work and stop the instance that owns the port using its own stop script. Launch from the desired release folder. |
| Model download fails | Check network access to the configured model source and retry. Preserve the expected checksum in the release configuration. |
| No Ollama models appear | Start Ollama, check the local API and `ollama list`, and refresh models in Settings. |
| The AI model exceeds available memory | Select a smaller installed vision model or continue with core screening while Ollama is offline. |

### Command line and developer checks

To process an approved recording from the project root, substitute its path below. Store the output in a fresh result directory. For synchronized views, supply their paths together after `--video`.

```powershell
.\.venv\Scripts\python.exe main.py process --video .\recording.mp4 --config .\config.yaml --output .\results\first-run
```

For development, install the test tools and run the checks.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
$env:ERGOPARTNER_TEST_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
node web/scripts/check-assessment-library.mjs
```
