# CoWTracker の uv 環境とツール（推論・データローダ・Rerun・TAP-Vid 評価）

AllTracker の fork（`demo.py` / `datasets/*` / `test_dense_on_sparse.py`）と同じ使い方を、CoWTracker でも
できるようにした追加分の説明です。対象は RGB 動画だけです（深度は使いません）。upstream のモデルコードへの変更は、
Blackwell GPU と 32 GB の VRAM で動かすための最小限だけで、どれも出力の数値は変えません（[変更点](#upstream-への変更点)）。

## セットアップ

```bash
git clone --recurse-submodules -b blackwell https://github.com/yuki-inaho/cowtracker.git
cd cowtracker
uv sync --locked            # Python 3.12, torch 2.7.0+cu128, xformers 0.0.30, timm 1.0.25, rerun-sdk 0.23.1
uv run pytest -q            # CPU だけ・重み不要のテスト
```

torch / torchvision / xformers は `https://download.pytorch.org/whl/cu128` から取ります（`pyproject.toml` の
`[tool.uv.sources]`）。CUDA 12.8 版は sm_120（RTX 50xx）を含みます。

### 重み

`--checkpoint` を省くと、Hugging Face の `facebook/cowtracker` から `cowtracker_model.pth`（3.9 GB、
CC-BY-NC-4.0）を取得します。ホーム以外に置く場合は、先に取得してパスを渡してください。

```bash
mkdir -p checkpoints
curl -L -o checkpoints/cowtracker_model.pth \
  https://huggingface.co/facebook/cowtracker/resolve/main/cowtracker_model.pth
```

`checkpoints/`、`*.pth`、`outputs/` は `.gitignore` の対象です。

### TAP-Vid（評価に使う場合）

```bash
mkdir -p <data>/tapvid && cd <data>/tapvid
curl -LO https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip          # 1.7 GB → tapvid_davis/tapvid_davis.pkl
curl -LO https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip   # 0.2 GB → tapvid_rgb_stacking/tapvid_rgb_stacking.pkl
unzip tapvid_davis.zip && unzip tapvid_rgb_stacking.zip
```

RoboTAP（`robotap/robotap.zip`、13.5 GB）と Kinetics（JPEG bytes の pkl）も、同じローダで読める形式です
（合成データのテストだけで確認しており、実データでは未評価です）。

## コマンド

### `cow-track`: 動画または画像フォルダの密な追跡

```bash
uv run cow-track --source videos/bmx-bumps.mp4 --checkpoint checkpoints/cowtracker_model.pth \
  --size 336 560 --out outputs/bmx --rrd --vram-limit-gb 30
uv run cow-track --source <画像フォルダ> --start 788 --end 888 --size 336 448 --out outputs/own --rrd
```

- `--source` には、動画（.mp4 .avi .mov .mkv .webm）か、画像フォルダ（.png .jpg .jpeg をファイル名順）を指定します。
  `--start` と `--end` は、ソート順での 0 始まりの位置で、end は含みません。範囲外を指定するとエラーになります。
- `--size H W`: H と W はどちらも 112 の倍数にします（[入力サイズ](#入力サイズ)）。16:9 前後の動画は 336 560、
  4:3 は 336 448 が目安です。
- `--mode auto`（既定）: `--window-len`（既定は upstream と同じ 100）以下のフレーム数なら単一パス、それを超えると
  upstream の `CoWTrackerWindowed`（窓 100、stride 100、メモリフレーム最大 10）を使います。
- `--device cuda`（既定）: CUDA が無ければエラーになります。CPU へ自動で切り替えることはありません。
  `--device cpu` は、重みを使わないテストのためのものです。CPU 推論は、実用上の選択肢として扱いません。
- `--dtype auto|fp16|bf16|fp32`（既定 auto = CUDA では fp16）: fp16 と fp32 の差は、見えている画素で平均 0.03 px です。
  bf16 は誤差が大きく、Turing では使えません。精度とメモリの詳細は [rtx2070_memory.md](rtx2070_memory.md) にあります。
- `--vram-limit-gb`: この値でプロセスの VRAM に上限をかけます。起動時の空きが上限より少なければ、開始しません。
- `--query-frame q`（既定 0）: 選んだフレームのうち q 番目の全画素を追跡します。AllTracker の demo と同じく、
  `video[q:]` を前向きに、`video[:q+1]` を逆順にして後ろ向きに推論し、重なるフレーム q は 1 回だけ使います。

出力（`--out`）:

| ファイル | 中身 |
| :--- | :--- |
| `tracks.npz` | `tracks` [T,H,W,2] float32（クエリフレームの全画素の (x, y) px）、`vis`・`conf` [T,H,W] float16、`frame_ids`、`size_hw`、`source_hw`、`query_frame` |
| `tracks.mp4` | `--grid-stride` 間隔の点を、位置に応じた色で描いた動画（vis×conf > `--vis-thr`、既定 0.1 = upstream の demo） |
| `tracks.rrd` | `--rrd` を付けたとき。Rerun 用（下記） |
| `meta.json` | 引数、呼び出しごとの mode、秒、peak VRAM、checkpoint の sha256、git commit / dirty、torch の版 |

### `cow-rerun`: Rerun の記録（.rrd）

```bash
uv run cow-rerun --tracks outputs/own/tracks.npz --source <画像フォルダ> --out outputs/own/tracks_stride8.rrd --grid-stride 8
uv run rerun outputs/own/tracks_stride8.rrd      # ビューア
```

`frame` の timeline に、`/frame/image`（JPEG）、`/frame/pred/points`、`/frame/pred/trails`（直近 `--trail`
フレームの軌跡）、`/metrics/pred/visible_fraction` を記録します。`--source` の frame_ids が npz と一致しない
場合はエラーになります。

### `cow-video`: 密なトラックの動画（AllTracker の demo 形式）

```bash
uv run cow-video --tracks outputs/own/tracks.npz --source <画像フォルダ> --out outputs/own/dense_rate2_hstack.mp4 \
  --rate 2 --stack h                         # 入力 | 追跡結果
uv run cow-video --tracks outputs/own/tracks.npz --source <画像フォルダ> --out outputs/own/dense_dots.mp4 \
  --rate 2 --stack h --bkg-opacity 0         # 黒背景に点だけ（彩度 x1.5）
```

AllTracker の `demo.py`（`--rate`、`--conf_thr`、`--bkg_opacity`、`--hstack` / `--vstack`）と同じ見た目の動画を作ります。

- クエリフレームの画素を `--rate` おきに（既定 2。336×448 なら 37,632 点）、各フレームで追跡した位置に描きます。
- 可視の判定は vis×conf > `--conf-thr`（既定 0.1）です。
- 色は、クエリフレームでの位置に応じた 2D カラーマップです。
- 点は rate に応じた半径の丸いアイコンで、GPU で splat します（AllTracker の `draw_pts_gpu` の移植）。
- `--stack h|v` で、入力を左か上に並べます。
- 出力は x264、`--crf 20`、yuv420p です。fps は source に従い、画像フォルダなら 10 です。

推論はしないので、同じ `tracks.npz` から、rate や背景を変えて何度でも作り直せます。

### `cow-eval-tapvid`: TAP-Vid 評価（δavg / AJ / OA）

```bash
uv run cow-eval-tapvid --pkl <data>/tapvid/tapvid_davis/tapvid_davis.pkl --dataset-name davis \
  --size 336 560 --out outputs/eval/davis_336x560 --checkpoint checkpoints/cowtracker_model.pth \
  --vram-limit-gb 30 --vis-thr-sweep 0.1 0.3 0.5 0.6 0.7 --render-videos 2
```

手順は AllTracker の `test_dense_on_sparse.py` と同じです。

- フレームは `--size` に resize し、GT は正規化座標 ×(W, H) にします。
- first モードです。各トラックのクエリは、最初に見えるフレームでの GT 位置です。
- ユニークなクエリフレーム q ごとに `video[q:]` を推論し、各クエリは最も近い画素の密トラックを読みます。
- vis×conf ≥ `--vis-thr`（既定 0.6）を可視とみなします。
- 指標は `(size-1)/255` で 256 スケールに換算し、動画ごとに求めてから平均します（×100）。

出力:

| ファイル | 中身 |
| :--- | :--- |
| `per_video.jsonl` | 逐次書き出し |
| `predictions/<動画>.npz` | tracks、visconf |
| `summary.json` | 平均、論文値、差、判定、閾値の掃引、provenance |
| `videos/<動画>.mp4` と `.rrd` | GT と予測の比較（`--render-videos` で指定した先頭 n 本） |

### `cow-track-sparse`: 多数の短いクリップで、指定した点だけを追跡

```bash
uv run cow-track-sparse --jobs <jobs.json> --out outputs/<名前> \
  --checkpoint checkpoints/cowtracker_model.pth --size 336 448 --vram-limit-gb 12
```

- モデルは 1 回だけ読み込み、`jobs.json` の各クリップを forward で 1 回ずつ推論します。
- 他のプロジェクト（例: RGB-D データの運動ラベル生成）から、別プロセスとして呼び出すためのコマンドです。
- CPU では動かしません（`--device cpu` はエラー）。`--checkpoint` は必須で、Hugging Face からの取得はしません。

`jobs.json` の形式（`schema` は `cow_sparse_jobs_v1`）:

| キー | 中身 |
| :--- | :--- |
| `id` | 出力ファイル名（`<id>.npz`）になる、パス区切りを含まない名前 |
| `image_dir` | 画像フォルダの絶対パス |
| `frames` | `image_dir` 直下のファイル名。時間順で、先頭が query frame。2 以上 `--window-len` 以下 |
| `queries` | query frame での元解像度の画素座標 `[[u, v], ...]`（画素中心は整数、サブ画素可） |

- 推論を始める前に、全クリップを検証します。検証でエラーになった場合は、出力を作りません。
- 追跡は `--size` に resize して行います。各 query は、推論座標 `u' = (u + 0.5)·W'/W − 0.5` で密な track 場から bilinear で読み、元解像度に戻します。

出力:

| ファイル | 中身 |
| :--- | :--- |
| `<id>.npz` | `uv` [N, T, 2]（元解像度の画素）、`visconf` [N, T]、`frames` |
| `manifest.json` | jobs の sha256、checkpoint と実行環境、attention、dtype、peak VRAM、ライセンス注記。最後に `status: "complete"` を書く |

出力は CoWTracker の結果なので、FAIR Noncommercial Research License の対象です（非商用の研究用途に限る）。

## Python API

```python
from cowtracker.toolkit.runner import build_runner
from cowtracker.toolkit.video_io import load_frames, resize_frames
from cowtracker.toolkit.tapvid import TapVidDataset
from cowtracker.toolkit.query_tracking import dense_tracks_from, track_queries_first

runner, checkpoint = build_runner("checkpoints/cowtracker_model.pth", "cuda", vram_limit_gb=30, window_len=100, mode="auto")
frames = load_frames("videos/bmx-bumps.mp4", max_frames=48)
dense = runner(resize_frames(frames.rgb, (336, 560)))      # DenseTracks: track [T,H,W,2], vis, conf, visconf
dense_mid = dense_tracks_from(runner, resize_frames(frames.rgb, (336, 560)), query_frame=24)  # 前後両方向
sample = TapVidDataset("tapvid_davis.pkl", (336, 560))[0]   # TapVidSample: video, points, occluded, query_points
tracks, visconf = track_queries_first(runner, sample.video, sample.query_points)
```

## Blackwell（RTX 5090）で分かったこと

### upstream への変更点

| 変更 | 理由 | 数値への影響 |
| :--- | :--- | :--- |
| `timm==1.0.25` に固定（`pyproject.toml`） | timm 1.0.26 以降は attention に `is_causal` を渡すが、upstream の `FlashAttention3.forward(x, attn_mask)` はそれを受け付けず `TypeError` になる | なし |
| `video_transformer` の attention の kernel: 9.x は FA3、8.0 以上は FA2、それ未満（Turing など）は cutlass。fp32 の入力は cutlass、CPU の入力は PyTorch の SDPA | xformers の FA3 は sm_90 専用で、sm_120 では `CUDA error: no kernel image` でプロセスが落ちる（`op=None` の自動選択も FA3 を選ぶ）。FA2 は sm_80 以上で fp16/bf16 専用なので、Turing と fp32 では `does not support inputs` になる | fp16/bf16 の Ampere 以降は変更なし。FA2 と SDPA の差は 0.0、cutlass と FA2 の差も fp32 基準の差の範囲（[rtx2070_memory.md](rtx2070_memory.md)） |
| runner は、CPU で dtype に変換してから GPU に移す | upstream の `from_checkpoint` は、fp32 の重み（3.6 GB）をいったん GPU に載せてから変換していた | なし（変換は同じ） |
| `CoWTracker.forward` / `CoWTrackerWindowed.forward`: feature 抽出の直後に `del tokens` | aggregator の全 24 層の出力（0.178 GB/フレーム）を、head の実行中も保持していた | なし（段ごとに呼んだ結果と `torch.equal` で一致） |
| `FeatureExtractor.drop_unused_layers`: aggregator の直後に、DPT が読まない 20 層を捨てる（読むのは 4, 11, 17, 23 層） | feature 抽出の段で、全 24 層を持ったまま DPT と ResNet を走らせていた | なし（同上） |
| `CoWTrackerWindowed.forward`: ループ末尾で `window_pred`、`extended_features`、`first_frame_features`、`frames` も捨てる | これらのビューが、前の窓の features（111 フレームで約 2.7 GB）を次の窓まで保持していた | なし（参照を捨てるだけ） |

これに加えて、toolkit の runner（`build_runner`）は、CUDA では `expandable_segments` を有効にし、各呼び出しの前に
`torch.cuda.empty_cache()` を呼びます。長さの違う suffix を続けて推論すると断片化で OOM になったためで、
これも数値には影響しません。

### 入力サイズ

H と W はどちらも 112（= lcm(14, 16)）の倍数が必要です。14 は ViT の patch、16 は ResNet の側枝の縮小率で、
側枝は 1/16 まで縮小してから ConvTranspose で戻します。

- 通った: 224×336、336×448、336×560、448×560、336×672
- 失敗した: 330×550、320×560、336×504、392×560、336×616（patch のアサーション、または 63 と 64 などの shape 不一致）

### VRAM（fp16、336×560）

| 条件 | peak（allocated） |
| :--- | :--- |
| 重み | 1.83 GB（974.5M パラメータ） |
| 1 フレームあたり | 0.207 GB（上の変更後。aggregator の段が上限。upstream のままだと 0.38 GB） |
| 100 フレーム（単一パス） | 22.9 GB / 8.3 s |
| 104 フレーム（windowed、DAVIS の cows 相当） | 23.6 GB / 9.3 s |
| 250 フレーム（windowed、RGB-Stacking 相当） | 26.3 GB（reserved 29.9 GB）/ 22.1 s |

upstream 既定の窓（100 フレーム + メモリ 10 フレーム + 先頭 1 フレーム）は、32 GB の GPU を専有すれば
`--vram-limit-gb 30` で動きます。他のプロセスと共有する場合は `--window-len` を小さくしてください。
ただし、窓の長さを変えると結果も変わります。

## 評価結果

2026-09-27 に RTX 5090 で実行しました。条件は、336×560、fp16、`--mode auto`（窓 100）、`--vis-thr 0.6`、上限 30 GB です。
数値は ×100 で、論文値は Table 1 の CoWTracker（Kubric で学習）です。

| データ | 本数 | AJ | δavg | OA | 論文 AJ / δavg / OA | 差 | 時間 / peak VRAM |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| TAP-Vid DAVIS（first） | 30 | 66.22 | 78.43 | 91.29 | 65.5 / 78.0 / 92.1 | +0.72 / +0.43 / −0.81 | 369 s / 23.6 GB |
| TAP-Vid RGB-Stacking（first） | 50 | 85.81 | 93.14 | 94.82 | 85.4 / 92.8 / 94.9 | +0.41 / +0.34 / −0.08 | 1123 s / 26.3 GB（全動画が windowed） |

- DAVIS も RGB-Stacking も、評価前に決めた基準（|ΔAJ| ≤ 2、|Δδavg| ≤ 1.5、|ΔOA| ≤ 2）の範囲に収まり、論文の数値を再現できたと判断しました。
  RGB-Stacking は 250 フレームの動画なので、windowed 推論（窓 100、メモリ 10 フレーム）の検証も兼ねています。
  δ_x（1, 2, 4, 8, 16 px）は 43.9 / 69.0 / 86.4 / 94.8 / 98.0 です。
- 可視の閾値を変えた場合（DAVIS の AJ / OA。δavg は閾値に依存しない）:

  | vis×conf の閾値 | AJ | OA |
  | :--- | ---: | ---: |
  | 0.1 | 63.25 | 89.81 |
  | 0.3 | 65.10 | 91.41 |
  | 0.5 | 65.98 | 91.48 |
  | 0.6 | 66.22 | 91.29 |
  | 0.7 | 66.16 | 90.67 |

- 論文の評価コードは公開されていないので、手順は AllTracker の評価に合わせています。GT の座標変換
  （×W か ×(W−1) か）、クエリ位置の丸め、可視の閾値などの細部は、論文とずれている可能性があります。
