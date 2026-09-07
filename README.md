# PartSR-RTSS26

This repository contains the PartSR paper method and one FullSR baseline.
Other comparison-method implementations are intentionally not included.

## Environment

Python 3.10 and uv 0.12 or newer are required. Choose one PyTorch variant:

```bash
uv sync --extra cpu
# or, on a CUDA 11.7-compatible GPU host:
uv sync --extra cu117
```

The `pyproject.toml` and `uv.lock` files are the authoritative dependency
definition. The project uses `opencv-contrib-python` because the tracker uses
`cv2.legacy.TrackerMOSSE_create`; do not install a second OpenCV wheel.

## External files

No model weights, videos, or FFmpeg binaries are stored in this repository.
Place the following files at the paths named by `config.yaml`:

- `super_resolution/models/EDSR_S_x4_0.8800.pth`
- `super_resolution/models/BasicVSR_ff_77000.pth`
- `extractor/best_rpn.pth`
- `models/classifier.pth`
- `scheduler/neuralucb7835.pth`

[Build the residual-enabled FFmpeg from `Yosame08/FFmpeg-Residual`](https://github.com/Yosame08/FFmpeg-Residual),
then set `ffmpeg_residual` in `config.yaml` to the resulting executable. The
tracked build recipe is
`tools/configure_h264_residual_probe.sh` in that repository; it uses static
linking and disables assembly implementations so the modified C decoder is
compiled into the binary.

RoI annotations are created and saved with the public
[semiauto-roi-labeler](https://github.com/Yosame08/semiauto-roi-labeler) tool.
The dataset preparation scripts convert its saved project JSON into the JSON
and per-frame YOLO labels expected by the PartSR runtime.

## Dataset

See [`Dataset_README.md`](./Dataset_README.md) for the video corpus, GT/LR pairs, and instructions for requesting access.

## Configuration

Set the paths and endpoints in `config.yaml`. The default dimensions are
`video_width=180`, `video_height=320`, and `chunk_len=3`. Set
`PARTSR_GPU_ID` to override `gpu_id` without editing the file.

The dataset root must contain DASH chunks and the matching 720p reference and
ROI labels, for example:

```text
server_folder/sell_3s/plant/chunk-stream0-00001.m4s
server_folder/sell_3s_720p/plant/720p-sell-plant.mp4
server_folder/sell_3s_720p/plant/plant_annotate/labels/*.txt
```

## Pipelines

Start the PartSR Edge server with:

```bash
uv run python server.py
```

Run the PartSR client testbed from another host (using an SSH tunnel or a
directly reachable Edge address):

```bash
PARTSR_DATA_ROOT=/path/to/server_folder uv run python testbed_basic.py
```

The FullSR baseline has separate entry points:

```bash
uv run python server_fullsr.py
uv run python testbed_fullsr.py \
  --data-root /path/to/server_folder \
  --source-dataset sell_3s \
  --dataset sell_3s_720p \
  --clip plant \
  --chunks 8 \
  --output-dir fullsr_results/plant
```

The FullSR testbed writes strict frame-count, RGB-order, SSIM, and timing
validation files. It is the only baseline included in this repository.

Training scripts and training datasets are intentionally outside this minimal
inference release; the existing parameter-retraining work is tracked in
`OPEN_SOURCE_TODO.md`.

## Validation status

The PartSR pipeline was validated across a 108 Edge host and an RTX 2080 Ti
client using eight 3-second chunks at 180x320 input resolution. The residual
protocol tests and scheduler tests are under `tests/`. Re-training NeuralUCB
and re-estimating residual-derived parameters for the deterministic H2RS v1
export remain documented in `OPEN_SOURCE_TODO.md`.
