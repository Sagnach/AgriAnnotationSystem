# AgriAnnotationSystem — Installation and User Guide

**Delivery method:** USB drive / local folder copy (no GitHub account required)  
**Target platform:** Windows 11, NVIDIA CUDA-capable GPU  
**Application:** AI-assisted agricultural image annotation (SAM 3 + human review)  
**Document version:** 1.0 — 27 October 2026

> **Important:** Copying the project directory does **not** automatically install Python, PyTorch, SAM 3, CUDA support, or the model checkpoint. Complete the setup below on the recipient computer. The instructions assume the supplied USB package contains the *complete* project, including the `third_party/sam3` source directory. Check the delivery manifest before beginning.

## 1. What this system does

The application detects, segments, and labels instances in agricultural images. The default example is apple orchard imagery. It offers **Fast** (whole-image) and **Deep Scan** (whole-image plus overlapping image tiles) modes, review of high- and lower-confidence candidates, and manual correction. Users can accept or delete predictions, change labels directly on the image, edit bounding boxes using SAM 3, add missing objects, refine masks via positive/negative clicks, and paint/erase mask pixels. Reviewed annotations can be exported in COCO instance-segmentation format.

This is a **human-reviewed pre-annotation tool**. Predictions must be checked before being treated as reliable annotations. The B3 version does not yet provide general Undo/Redo.

## 2. Hardware and software prerequisites

| Requirement | Recommended / tested configuration | Notes |
|---|---|---|
| Operating system | Windows 11, 64-bit | This guide is Windows-specific. |
| Python | **Python 3.12, 64-bit** | Development environment used **3.12.10**. Avoid installing Python 3.13 for this delivery without new tests. |
| GPU | NVIDIA GPU with sufficient VRAM | Development verified using RTX 5080 Laptop, 16 GB VRAM. Around 4–5 GB of GPU memory was allocated in representative experiments; allow extra headroom. Other GPUs must be tested. |
| GPU drivers | Current NVIDIA driver supporting the selected PyTorch CUDA build | `nvidia-smi` should detect the GPU. |
| PyTorch | **CUDA-enabled**, not CPU-only | Development used `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`. Install the matching wheel available from the official PyTorch installer. |
| CUDA | CUDA **12.8 PyTorch build** for the reproduced development environment | The matching PyTorch wheel includes its CUDA runtime components; a full stand-alone CUDA Toolkit is not automatically required. |
| Browser | Current Chrome or Edge | Gradio UI runs locally at `http://127.0.0.1:7860`. |
| Disk space | Several GB free, beyond the USB package | Required for dependencies, model checkpoint and outputs. |
| Internet | Needed for normal online installation and optional checkpoint retrieval | Offline installation needs pre-collected wheels **and a local model checkpoint**. |

If the receiving machine does not have an NVIDIA CUDA GPU, the application has a CPU option, but this configuration has not been verified for practical SAM 3 use and may be very slow.

## 3. USB delivery contents — check before installation

Copy the entire `AgriAnnotationSystem` directory from the USB drive to a **local** drive, for example:

```text
D:\AgriAnnotationSystem\
```

Do not run the project directly from a removable USB drive; local disk execution avoids file access and write issues. The recipient folder may use any drive letter: the app resolves its project root from the script location.

Expected essential files:

```text
AgriAnnotationSystem/
├── README.md                     <-- this guide (rename on delivery)
├── requirements.txt     <-- direct application dependencies
├── scripts/
│   └── 05_launch_annotation_app.py
├── src/
│   ├── annotation/annotation_manager.py
│   └── ui/interactive_canvas.py
├── third_party/
│   └── sam3/
│       ├── pyproject.toml
│       └── sam3/                   <-- complete official SAM 3 Python source
├── models/
│   └── sam3/checkpoints/
│       └── sam3.pt                <-- REQUIRED for fully offline inference
├── configs/                       <-- where supplied
├── data/                          <-- sample images/dataset if supplied
├── workspace/                     <-- writable user annotation projects
└── outputs/                       <-- optional development experiment reports
```

The above tree is an **expected delivery checklist**, not a claim that every item is already present on the USB drive. The person preparing the USB must verify it. Particularly:

- `third_party/sam3/` must contain the actual SAM 3 source code, not only an empty folder.
- The *original development PC's* Hugging Face model cache usually lies **outside** this project folder. A working developer installation does **not** prove `models/sam3/checkpoints/sam3.pt` is packaged.
- The `.venv` / `.venv312` directory from a different PC should **not** be copied as the installation method. Create a new virtual environment on the recipient PC.
- Remove private tokens and any unapproved/private images before delivery. Only redistribute model/data files in accordance with their respective licenses and access terms; recipients may need their own checkpoint access permission.

## 4. Online setup: step by step (recommended)

Open **PowerShell** in Windows.

### Step 1 — install Python 3.12

Install 64-bit Python 3.12 from <https://www.python.org/downloads/>. On installation, enable the Python Launcher (`py`) and, if offered, Add Python to PATH. Then verify:

```powershell
py -3.12 --version
```

Expected: `Python 3.12.x`.

### Step 2 — open the copied project and create an isolated virtual environment

```powershell
cd D:\AgriAnnotationSystem
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

All remaining commands use `\.venv\Scripts\python.exe` explicitly. **No environment activation is required**, avoiding PowerShell script-execution-policy problems.

### Step 3 — check the NVIDIA GPU

```powershell
nvidia-smi
```

If this fails, install/update the appropriate NVIDIA driver and restart Windows. Do not proceed with a CPU-only PyTorch installation by accident.

### Step 4 — install CUDA-enabled PyTorch and torchvision

The development environment was `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128`. Use the official PyTorch selector at <https://pytorch.org/get-started/locally/> to obtain its currently supported matching **Windows / pip / Python / CUDA** command.

To reproduce the tested **CUDA 12.8** installation *when compatible wheels for these versions are available*, use:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

If these precise wheels are not available for the recipient's machine, do **not** silently substitute a CPU build; select a compatible CUDA build from the official selector and recheck compatibility with Windows Triton (next step). Record installed versions for the handover.

Verify:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('torch:',torch.__version__); print('CUDA build:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Proceed only after `CUDA available: True` on the recommended NVIDIA setup.

### Step 5 — install SAM 3 from the USB's local source copy

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\third_party\sam3
```

This **does not clone GitHub**. It installs the SAM 3 Python package using the source files already copied from the USB. The command may still download SAM 3 package dependencies from PyPI unless an offline wheelhouse is supplied.

### Step 6 — install the application and SAM 3 additional dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The delivery requirements intentionally include dependencies that were needed by the working Windows SAM 3 import path but were not fully resolved by the base SAM 3 installation: **`einops`, `pycocotools`, `psutil`, and Windows Triton**. The file also includes Gradio 6 and core image processing packages.

**Important:** The provided Windows Triton pin (`triton-windows==3.6.0.post26`) reflects the development setup based on PyTorch 2.11. If a different torch version is installed, consult the compatibility table at <https://github.com/triton-lang/triton-windows> before changing Triton.

Check dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import sam3, triton, einops, psutil, pycocotools, gradio, cv2, numpy, PIL; print('SAM3 IMPORT PASS'); print('Gradio:',gradio.__version__); print('Triton:',triton.__version__)"
```

A `FutureWarning` concerning `timm.models.layers` is not usually an installation failure. An actual `ModuleNotFoundError`, CUDA error, or `pip check` failure must be resolved.

### Step 7 — make the SAM 3 model checkpoint available

**Preferred for USB handover:** The authorised provider places the checkpoint at:

```text
D:\AgriAnnotationSystem\models\sam3\checkpoints\sam3.pt
```

Confirm that it is a real model checkpoint and not a placeholder/empty file. Do not rename an arbitrary weights file to `sam3.pt`.

**Alternative if the USB does not include weights:** The receiving user must obtain the appropriate SAM 3 checkpoint permissions at <https://huggingface.co/facebook/sam3> and download it while logged in. If the official build supports automatic downloads in the installed SAM 3 revision, the app can use that route. Install and authenticate Hugging Face CLI if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -U huggingface_hub
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\hf.exe auth whoami
```

The checkpoint and licensing conditions are independent of having the SAM 3 source code. **For a reliable offline USB delivery, verify the recipient is authorised to receive and use the checkpoint, and test with `--no-hf-download`.**

### Step 8 — run the application

```powershell
cd D:\AgriAnnotationSystem
.\.venv\Scripts\python.exe scripts\05_launch_annotation_app.py --no-hf-download
```

The above command **requires a local checkpoint**. If the file was not included and the recipient is authorised to download one, omit `--no-hf-download` after logging in:

```powershell
.\.venv\Scripts\python.exe scripts\05_launch_annotation_app.py
```

Open <http://127.0.0.1:7860> in a browser (normally opens automatically). Keep the PowerShell window running while using the app. Stop the server using **Ctrl+C**.

If the port is already occupied, run:

```powershell
.\.venv\Scripts\python.exe scripts\05_launch_annotation_app.py --no-hf-download --port 7861
```

Then open <http://127.0.0.1:7861>.

## 5. How to use the annotation system

1. Select **Load SAM 3** to initialise the model (GPU loading may take several seconds).
2. Upload an agricultural image. For an apple orchard, use the prompt `orchard apple` and class label `apple`.
3. Choose **Fast** for routine images or **Deep Scan** for small/dense/partially visible fruit. Deep Scan takes longer.
4. Run automatic annotation; inspect coloured masks, boxes, class labels, and confidence scores. Optionally display lower-confidence candidate instances.
5. Review each object: **Accept / Delete / Relabel**; use **Add Missing** for missed fruit and **Refine (+/−)** for point-guided mask correction.
6. **B1:** click an image label to edit the class inline. **B2:** select an object, edit its box with drag handles, then Apply; SAM 3 regenerates its mask. **B3:** use Brush Mask, Add Pixels / Erase Pixels, choose brush size, paint and Apply (or Cancel).
7. Save the reviewed annotations and export COCO JSON. Confirm your reviewed project files are present under `workspace/projects/`.

**Do not close the browser or terminal before saving.** Review the exported classes and masks before using them to train another model. A box edit may produce a final rectangle different from the hand-dragged prompt because the stored bounding box is recalculated from the new segmentation mask.

## 6. Offline installation (for an Internet-restricted recipient)

An **offline deployment** needs more than copying the Python source. The sender must prepare, on a compatible Windows x64 + Python 3.12 environment:

- A Python 3.12 installer (redistribution subject to its license).
- A `wheelhouse/` containing compatible wheels **and all transitive dependencies** for CUDA PyTorch, `torchvision`, the SAM 3 source requirements, and `requirements.txt` (including `triton-windows`).
- The **complete** SAM 3 source directory and any additional assets required by the included revision.
- An authorised, locally loadable `models/sam3/checkpoints/sam3.pt` (and any ancillary checkpoint assets needed by that revision).
- The application source, README, sample image, and an export test.

A possible online *preparation* step on the sender's compatible Windows computer is:

```powershell
py -3.12 -m pip download -r requirements.txt -d wheelhouse
```

This command **alone is not sufficient** to bundle CUDA PyTorch or all SAM 3 build dependencies. The sender must also collect the selected CUDA torch/torchvision wheels, SAM 3 build dependencies, and verify the **entire** installation with network disabled. Pip editable installs can trigger build-isolation dependency retrieval unless the build dependencies are already installed locally.

On the recipient computer, the offline application packages may then be installed with commands of this form **after** Python, pip, the compatible CUDA wheels, and build requirements are in place:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links .\wheelhouse -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-index --no-build-isolation -e .\third_party\sam3
```

The exact offline sequence depends on the complete wheel inventory and must be validated by the person preparing the USB. An offline install should not be advertised as ready until it has passed a clean-machine/no-network trial.

## 7. Troubleshooting

| Symptom | Likely cause | Corrective action |
|---|---|---|
| `No module named 'sam3'` | Official source not installed or wrong interpreter | Check `third_party/sam3`, then use `.venv\Scripts\python.exe -m pip install -e .\third_party\sam3`. |
| `No module named 'einops'` / `pycocotools` / `psutil` | Missing runtime dependencies | Run `pip install -r requirements.txt` using the `.venv` interpreter. |
| `No module named 'triton'` | Windows Triton wheel missing | Install the `triton-windows` version compatible with installed PyTorch. |
| `torch.cuda.is_available()` is `False` | Missing driver, no NVIDIA GPU, or CPU PyTorch wheel | Check `nvidia-smi`, installed torch/CUDA build, and GPU support. |
| CUDA / Triton kernel error | CUDA/PyTorch/Triton/driver mismatch | Reconcile the version matrix; avoid arbitrary upgrades. |
| Model checkpoint missing / HTTP 401/403 | Checkpoint not included or gated access not approved | Verify `models/sam3/checkpoints/sam3.pt` or obtain authorised access and login. |
| Browser page does not open | Server stopped / wrong address / blocked port | Check terminal, use `http://127.0.0.1:7860`, or change `--port`. |
| Gradio custom HTML event error | Incompatible Gradio major version | Install Gradio `>=6,<7` and restart the app. |
| Edit appears changed but not exported | Review not saved / export not updated | Use the Save/Export control, reopen COCO JSON and verify. |
| Running slowly or out of GPU memory | Deep Scan too costly for available GPU | Try Fast, close other GPU programs, and use supported hardware. |

## 8. Acceptance checklist for the receiver

- [ ] Project copied to a **local** directory; essential source files are present.
- [ ] Python 3.12 x64 and a new virtual environment are installed.
- [ ] `pip check` reports no broken requirements.
- [ ] CUDA-capable PyTorch recognises the NVIDIA GPU.
- [ ] `import sam3` succeeds.
- [ ] A legitimately acquired SAM 3 checkpoint loads (offline if required).
- [ ] The Gradio UI opens locally and automatically annotates a supplied sample image.
- [ ] Fast and Deep Scan can run; candidates are displayed with labels and masks.
- [ ] B1 inline label edit, B2 bounding-box correction, and B3 brush add/erase work.
- [ ] Reviewed annotations save and exported COCO JSON contains the corrected class and segmentation geometry.
- [ ] Application can be stopped and restarted without reinstalling dependencies.

## 9. Support and reproducibility

When reporting a problem, provide the full error message, Windows version, GPU name, and these command outputs:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip freeze > installed_packages.txt
nvidia-smi
```

The sender should retain a matching `installed_packages.txt` (from the **working development environment**) and the exact SAM 3 source revision. `requirements.txt` lists direct/known runtime requirements; it is **not** a byte-for-byte lockfile of every transitive dependency. To reproduce the precise development environment, use the exported version inventory and ensure the recipient hardware is compatible.

### Third-party notices

SAM 3 code/checkpoint, PyTorch, MinneApple, and other dependencies remain subject to their respective terms. Copy required attribution and licence files with the USB delivery and confirm redistribution permission for any gated model weights or dataset files.
