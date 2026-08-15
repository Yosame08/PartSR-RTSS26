# SR Video Dataset

This directory contains three video categories: live commerce, sports, and gaming. The data can be used for video super-resolution (VSR) and related tasks.

## Data Availability

The dataset used in this work is currently hosted on a local server maintained by our research group and is not yet publicly downloadable. We are evaluating two options for broader distribution: (1) hosting a mirrored copy on a public cloud storage service, and (2) providing controlled remote access to the local server via VPN with appropriate security restrictions.

In the meantime, researchers interested in obtaining the dataset are welcome to contact us directly, and we will respond with access instructions:

- Yuchen Wang — wang_yuchen@connect.hku.hk
- Weijia Lang — 24210240192@m.fudan.edu.cn

Please include your name, affiliation, and intended use of the dataset in your request.

## Video Statistics

Scope: `.mp4` files in the three directories listed below. Statistics were collected on August 4, 2026.

| Scene | Video directory | Count | Total duration | Duration per video | Resolution | Frame rate | Codec | Size |
|---|---|---:|---:|---:|---|---:|---|---:|
| Live commerce | `datasets_prepare/videos` | 66 | 00:46:41 | 15.6-105.4 s | 720 x 1280 | 30 fps | H.264 | 2.668 GiB |
| Sports | `Sports-dataset/extracted_videos` | 3,200 | 18:34:22.6 | 2.0-147.0 s | 1280 x 720 | 25 fps | MPEG-4 | 60.598 GiB |
| Gaming | `Game-dataset` | 41 | 01:02:37.8 | 89.5-120.1 s | 1280 x 720 | 30 fps | H.264 | 1.168 GiB |
| **Total** | - | **3,307** | **20:23:41.8** | - | - | - | - | **64.434 GiB** |

> Note: Not all videos are 30 fps. The live-commerce videos are portrait-oriented at `720 x 1280`, while the sports videos are `25 fps`. These properties were verified from the video metadata.

## Scene Composition

- **Live commerce:** `datasets_prepare/videos` contains the source MP4 files, while `Sell-dataset` contains the corresponding GT/LR frames.
- **Sports:** `Sports-dataset/extracted_videos` contains basketball, football, volleyball, and aerobic gymnastics videos, with 800 videos per sport split across `trainval` and `test`.
- **Gaming:** `Game-dataset` contains Elden Ring (12 videos), LOL (17 videos, including 6 in `LOL_test`), and VALORANT (12 videos).

## GT/LR Data

GT contains high-resolution frames, and LR contains their low-resolution counterparts. Each GT/LR pair uses the same directory structure and frame filenames.

| Scene | Video frame sequences | Image subset |
|---|---|---|
| Live commerce | `video_7_11`: 554 sequences x 60 frames; `video_8_20`: 474 sequences x 90 frames | `pic_7_11`: 2,314 GT images and 2,314 LR images |
| Sports | `sports_set_gt` / `sports_set_lr`: 2,116 sequences and 158,700 frames in each set | `sports_sr_pic_subset`: 19,252 GT images and 19,252 LR images |
| Gaming | `game_set_gt` / `game_set_lr`: 221 sequences and 19,890 frames in each set | `game_sr_pic_subset`: 7,576 GT images and 7,576 LR images |

The sports and gaming data also include `labels` directories. Sports metadata is available under `MultiSports`, and the current gaming annotations contain six LOL JSON files.
