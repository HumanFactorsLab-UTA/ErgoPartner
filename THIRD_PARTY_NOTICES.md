# Third-party notices

The evaluation license covers only rights controlled by the project rights holders.
The bundled Space Grotesk font is by Florian Karsten and uses the SIL Open Font License 1.1. [Font license](web/public/brand/SpaceGrotesk-OFL.txt)

The web build includes a patched copy of picomatch under MIT. The source, original notice, upstream integrity information and patch are supplied together. [License](web/vendor/picomatch/LICENSE) · [Patch record](web/vendor/picomatch/PATCH.md)

Python and JavaScript dependencies are installed separately. Their distributions retain their respective license and attribution files. The web lockfile identifies the resolved package versions. [Python dependencies](requirements.txt) · [Web lockfile](web/package-lock.json)

## Models and external tools

The MediaPipe library uses Apache-2.0. Google's pose and hand model cards also
identify Apache-2.0. ErgoPartner downloads the model artifacts from Google's
distribution service and verifies the configured SHA-256 values.

- Pose model card, page 2, identifies the model license.
  [BlazePose GHUM 3D model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf)
- Hand model card, page 2, identifies the model license.
  [Hand Tracking model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf)
- The library license includes additional component notices.
  [MediaPipe license](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)

FFmpeg is installed separately. Its license depends on the chosen build and
enabled components. Retain the notices and source provisions supplied with the
specific binary distribution. [FFmpeg licensing](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md)

The PDF renderer uses Playwright under Apache-2.0 and downloads a matching
Chromium browser. Chromium and its bundled components have their own notices
within that distribution. [Playwright license](https://github.com/microsoft/playwright/blob/main/LICENSE),
[Chromium license](https://chromium.googlesource.com/chromium/src/+/HEAD/LICENSE)

The optional Ollama service uses MIT. The configured Gemma 4 model family uses
Apache-2.0 according to its provider. Review the specific model's terms when
choosing another tag. [Ollama license](https://github.com/ollama/ollama/blob/main/LICENSE),
[Gemma 4 license](https://gemma.ai/resources/apache-2/)

## Scientific attribution

The RULA, REBA and RNLE sources, authors, publication details and method versions
are preserved in ERGONOMICS_METHODS.md. These citations credit the methods used by
the implementation. They do not imply endorsement by the authors or publishers.
