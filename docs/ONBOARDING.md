# LLMオンボーディングサマリー（CoWTracker / `blackwell` ブランチ）

> 新しいエージェントや開発者が、このリポジトリだけを見て、環境・契約・検証の手順をつかむための入口です。
> コマンドはすべてリポジトリのルートで実行します。使い方の詳細は [uv_tools.md](uv_tools.md) にあります。

> `rtx2070` ブランチの実機セットアップは [rtx2070_setup.md](rtx2070_setup.md) を参照してください。
> 以下の Blackwell 評価と RTX 2070 の実機確認は条件が異なります。

**更新:** 2026-09-27
**対象:** `yuki-inaho/cowtracker` の `blackwell` ブランチ（upstream は `facebookresearch/cowtracker`）

## 1. プロジェクト概要と目的

- **名称と領域:** CoWTracker は、密な点追跡のモデルです。VGGT の backbone と、warping による反復 refinement を組み合わせており、
  コストボリュームを使いません。入力は RGB 動画で、出力は、フレーム 0 の全画素について、各フレームでの位置・可視性・信頼度です。
- **この fork の成果物:**
  - RTX 50xx（Blackwell、sm_120）で動く、lock つきの uv 環境。
  - `cowtracker/toolkit` と、4 つのコマンド（`cow-track`、`cow-rerun`、`cow-video`、`cow-eval-tapvid`）。
  - テスト（CPU、重み不要）と、GPU の E2E テスト。
  - 文書: この文書と `uv_tools.md`。
- **目的:** AllTracker の fork と同じ感覚で、推論、データローダ（動画・画像フォルダ・TAP-Vid pkl）、Rerun 出力、
  動画評価（TAP-Vid の AJ / δavg / OA）を使えるようにすること。対象は RGB だけで、深度は扱いません。
- **確認済み**（2026-09-27、RTX 5090 32 GB、fp16、336×560、`--vis-thr 0.6`）:
  - TAP-Vid DAVIS（30 本）: AJ / δavg / OA = 66.22 / 78.43 / 91.29（論文 65.5 / 78.0 / 92.1）
  - TAP-Vid RGB-Stacking（50 本、windowed 推論）: 85.81 / 93.14 / 94.82（論文 85.4 / 92.8 / 94.9）
  - テスト: CPU 58 件、GPU 4 件が成功。
- **未検証:**
  - Kinetics と RoboTAP の実データ（ローダの形式は、合成データのテストでだけ確認しています）。
  - Hopper など他の GPU（FA3 を使う経路は upstream のままで、ここでは動かしていません）。
  - Blackwell 評価と同条件での RTX 2070 実機評価（[rtx2070_memory.md](rtx2070_memory.md)）。
    `rtx2070` ブランチでは、別条件の短い実データによる動作確認を実施済み（[rtx2070_setup.md](rtx2070_setup.md)）。
  - CPU 推論は選択肢にしない（CPU の経路は、重みを使わないテストのためのもの）。
  - 学習と fine-tune（upstream にも学習コードはありません）。

## 2. クリティカルな要求・制約

| 境界 | 守る契約 |
| :--- | :--- |
| 入力 | RGB の uint8 [T,H,W,3]。H と W はどちらも **112 の倍数**（patch 14 × ResNet の側枝 1/16）。resize は `--size` で明示し、自動で丸めない |
| 出力 | `track` [T,H,W,2] は、クエリフレーム（既定 0）の各画素の (x, y) px。画素中心が整数座標の規約で、クエリフレームは恒等。`vis` と `conf` は [0,1]。q > 0 では前後両方向に推論する |
| 可視判定 | 表示は vis×conf > 0.1（upstream の demo）、評価は vis×conf ≥ 0.6（AllTracker の評価）。混同しない |
| 推論経路 | `--mode auto`: T ≤ `--window-len`（既定 100 = upstream）なら単一パス、超えたら upstream の `CoWTrackerWindowed`（窓 100、stride 100、メモリ 10 フレーム） |
| VRAM | 重み 1.83 GB + 約 0.207 GB/フレーム（336×560、fp16）。全 GPU ジョブに `--vram-limit-gb` を明示する。OOM のときに窓や解像度を黙って変えない |
| fallback 禁止 | CUDA が無いのに `--device cuda`、未対応の入力、範囲外のフレーム、112 の倍数でないサイズ、空き VRAM の不足、pkl の形式違反は、すべて明示エラーにする |
| upstream の変更 | 次の 4 件だけで、どれも出力の数値は変えない。戻すと sm_120 でのクラッシュか OOM が再発する（詳細は [uv_tools.md](uv_tools.md)）。<br>・timm の固定<br>・FA3 を Hopper 専用にする<br>・tokens と DPT 未使用層の早期解放<br>・windowed での参照の解放 |
| 依存の固定 | torch 2.7.0+cu128、xformers 0.0.30、**timm==1.0.25**（1.0.26 以降は `is_causal` の TypeError になる）。上げる場合は GPU テストと TAP-Vid の再評価をしてから |
| ライセンス | コードは FAIR Noncommercial Research License（`LICENSE`）。重みは CC-BY-NC-4.0。TAP-Vid の注釈は CC BY。メトリクスは AllTracker（MIT）経由の tapnet（Apache-2.0）の移植で、出典はファイル冒頭 |
| 公開の境界 | 次は commit しない。<br>・`temp/`、`outputs/`、`checkpoints/`、`*.pth`<br>・データセット本体<br>・実機の絶対パス<br>・非公開データの名前や機材の識別子<br>・作業ログ |

## 3. 参照すべき合意済み資料

| 種別 | ファイル | 用途 |
| :--- | :--- | :--- |
| 入口 | `README.md` | upstream の説明と、この fork の節（uv 環境とツール） |
| 利用方法 | [uv_tools.md](uv_tools.md) | セットアップ、コマンド、出力の形式、評価プロトコル、upstream の変更点、VRAM の表、評価結果 |
| 精度と小さな GPU | [rtx2070_memory.md](rtx2070_memory.md) | fp16 / bf16 / fp32 の精度、kernel の選び方、メモリのモデル式、RTX 2070 の模擬と推奨設定 |
| 論文 | [cowtracker.pdf](cowtracker.pdf) | Table 1（TAP-Vid の数値）、入力解像度 336×560、K=5 回の反復 |
| 環境 | `pyproject.toml`、`uv.lock` | 依存の固定、cu128 の index、console scripts、pytest の marker `gpu`、ruff の設定 |
| 実装 | `cowtracker/toolkit/` | 下の表 |
| テスト | `tests/` | 入出力、メトリクス、クエリ追跡、Rerun の読み戻し、CLI、失敗の経路（CPU）。`test_gpu_e2e.py` は実重み |
| 作業書 | `temp/`（clone には含まれない） | 手元の計画と記録。無ければ、本書から新しく作る。公開に必須の資料ではない |

`cowtracker/toolkit/` の責務:

| モジュール | 責務 |
| :--- | :--- |
| `video_io.py` | 動画と画像フォルダの読み込み（ファイル名順、0 始まりの位置、end は含まない）、bilinear の resize |
| `tapvid.py` | TAP-Vid の pkl（dict / list、生の配列 / JPEG bytes）→ `TapVidSample`（first モードのクエリ） |
| `tracks.py` | `DenseTracks`（track、vis、conf、visconf）と `DenseTracker` の Protocol |
| `runner.py` | サイズの規則、mode の規則、`CowTrackerRunner`（重みは 1 回だけロード、fp16 の autocast）、`build_runner` |
| `runtime.py` | device の明示、VRAM の上限と空きの検査、expandable segments、peak VRAM、git / torch のメタデータ、sha256 |
| `query_tracking.py` | `dense_tracks_from`（クエリフレームから前後両方向）と、疎なクエリを密なトラッカーで追跡する処理（ユニークなクエリフレームごとに suffix を推論） |
| `metrics.py` | TAP-Vid のメトリクス（256 スケールへの換算） |
| `render.py` / `rerun_log.py` | AllTracker 形式の splat 描画（`splat_tracks`: rate、背景の不透明度、彩度）と Rerun の記録 |
| `track_cli.py` / `rerun_cli.py` / `video_cli.py` / `eval_cli.py` | 4 つのコマンド。`track_cli` と `eval_cli` は `main(argv, tracker=None)` で tracker を注入できる |

## 4. タスク境界（任せること / 任せないこと）

### 任せるタスク（例）

- 入力形式の追加（例: 別のデータセットのローダ）、出力の追加（例: 別の可視化）、CLI の堅牢化と、それに合わせたテストの追加。
- 明示した条件での TAP-Vid の再評価。条件、所要時間、peak VRAM、判定を記録する。
- 数値を変えないメモリや速度の改善。段ごとに呼んだ結果との `torch.equal` と、peak の実測で確かめる。
- 依頼された文書の更新と commit / push。stage する前に、下の private 情報の検査を通す。

### 任せないタスク（例）

- 評価プロトコル（クエリの作り方、座標の変換、閾値 0.6）を黙って変えること。変える場合は、別のオプションとして追加する。
- 窓の長さや解像度を変えた結果を、論文の条件の結果として報告すること。
- 無指示でのデータ・重みの公開、大きな生成物の commit、force push、他の作業者のプロセスや出力の削除。
- テストを通すために、サイズの検査やエラーの経路を緩めること。

## 5. インタラクション方針

- 日本語で簡潔に報告し、「結論 → 根拠（コマンドと数値）→ 残る課題」の順に書く。時間のかかる処理は途中経過を伝える。
- 「確認済み」「未検証」「推定」を区別する。条件の違う測定を並べて、改善を主張しない。
- GPU を他のエージェントと共有しているかどうかは、作業の初めに確認する。共有している場合は上限を下げ、大きな予算（例: 20 GB 以上）が必要なら相談する。
- 公開する文書では、相対パスとプレースホルダ（`<data>`、`<画像フォルダ>` など）を使う。生のログは `temp/` に置く。

## 6. 試行タスク（オンボーディング演習）

### 演習 1: 環境とテスト

```bash
git status --short --branch
uv sync --locked
uv run python -c "import torch, xformers, timm; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(), xformers.__version__, timm.__version__)"
uv run pytest -q
uv run ruff check cowtracker/toolkit tests && uv run ruff format --check cowtracker/toolkit tests
```

期待: `2.7.0+cu128 True (12, 0) 0.0.30 1.0.25`（RTX 50xx の場合）、CPU のテストはすべて成功。

### 演習 2: 推論と Rerun

```bash
mkdir -p checkpoints
curl -L -o checkpoints/cowtracker_model.pth https://huggingface.co/facebook/cowtracker/resolve/main/cowtracker_model.pth
export COWTRACKER_CHECKPOINT=checkpoints/cowtracker_model.pth
uv run pytest -m gpu -q
uv run cow-track --source videos/bmx-bumps.mp4 --checkpoint $COWTRACKER_CHECKPOINT --size 336 560 \
  --out outputs/bmx --rrd --vram-limit-gb 30
uv run rerun outputs/bmx/tracks.rrd          # 画面が無い環境では --serve-web
uv run cow-video --tracks outputs/bmx/tracks.npz --source videos/bmx-bumps.mp4 \
  --out outputs/bmx/dense_rate2_hstack.mp4 --rate 2 --stack h   # AllTracker 形式の密なトラック動画
```

期待（RTX 5090）: 90 フレームが単一パスで数秒。`meta.json` に mode、peak VRAM（約 23〜25 GB）、checkpoint の sha256 が入る。
`tracks.npz` のフレーム 0 は恒等になる。

### 演習 3: TAP-Vid を 1 本だけ評価して読む

```bash
uv run cow-eval-tapvid --pkl <data>/tapvid/tapvid_davis/tapvid_davis.pkl --dataset-name davis --size 336 560 \
  --max-videos 1 --out outputs/eval/davis_probe --checkpoint $COWTRACKER_CHECKPOINT --vram-limit-gb 30
```

`summary.json` の `mean`、`paper`、`delta`、`verdict` と、`per_video.jsonl` の `query_frames`、`calls` を説明できることを確認します。
1 本だけの値は、論文の平均とは比べません（全 30 本で比べる）。

## 7. 運用ルール・変更管理

- **品質ゲート（commit の前）:** `uv run pytest -q`、`uv run ruff check cowtracker/toolkit tests`、
  `uv run ruff format --check cowtracker/toolkit tests`。モデル側に触れたときは `uv run pytest -m gpu -q` も通す。
- **TDD:** 先に失敗するテストを書き、Red を確認してから実装する。メモリの修正では、修正前後の peak の実測を
  テストのコメントに残す。
- **private 情報の検査（stage した後）:** 追加行に、絶対パスとメールアドレスが無いこと（ヒット 0）を確かめる。
  ```bash
  git diff --cached | sed -n 's/^+//p' | grep -v '^++' \
    | grep -nE '(^|[^A-Za-z0-9_.:/-])/(home|Users|root|mnt|media|data|workspace|tmp|opt|srv)/|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}'
  ```
  非公開データの名前や機材の識別子は機械的に検出できないので、`git diff --cached --stat` と差分を目で確認する。
  `git add -A` は使わず、ファイルを明示して stage する。
- **結果の来歴:** CLI は `meta.json` / `summary.json` に git の commit と dirty を記録する。
  未 commit の状態で評価する場合は、`git diff` を `outputs/` に保存して、結果と対応づける。
- **作業書:** 大きな作業は `temp/workdoc_<Mon><DD>-<YYYY>_<slug>.md` に計画と記録を書く（gitignore 対象）。
  手順は write → review → 実行。
- **生成物:** `outputs/` の下（gitignore 対象）。重みは `checkpoints/`（同じく gitignore 対象）。

## 8. トラブルシューティング

| 症状 | 原因と対処 |
| :--- | :--- |
| `CUDA error: no kernel image is available`（flash_fwd_launch_template） | FA3 の kernel は sm_90 専用。`video_transformer._get_flash_attention_ops` が capability 9 のときだけ FA3 を使っているか確認する。`op=None` の自動選択も FA3 を選ぶので使わない |
| `FlashAttention3.forward() got an unexpected keyword argument 'is_causal'` | timm が 1.0.26 以降になっている。`uv sync --locked` で 1.0.25 に戻す |
| `memory_efficient_attention does not support inputs` | FA2 に、fp32 か sm_80 未満の GPU の入力が渡っている。このブランチの `video_transformer` は cutlass を選ぶので、古いコードを使っていないか確認する |
| RTX 2070 など 8 GB の GPU | [rtx2070_memory.md](rtx2070_memory.md) の推奨設定（fp16、解像度と `--window-len`、`--vram-limit-gb 7`）に従う |
| `Input image height ... is not a multiple of patch height 14`、`size of tensor a (63) must match ... (64)` | 入力サイズが 112 の倍数でない。`--size` を 336 560 や 336 448 などにする |
| CUDA OOM | ①`reserved but unallocated` が大きいなら断片化。runner は expandable segments と `empty_cache` を使うので、runner を通さずに直接 model を呼んでいないか確認する。②本当に足りないなら、上限・フレーム数・解像度を確認する。窓や解像度を下げると結果が変わるので、そのことを記録する |
| `free VRAM ... is below the requested limit` | 他のプロセスが GPU を使っている。上限を下げるか、空くのを待つ（自動では待たない） |
| 重みを取得する場所 | `--checkpoint` を省くと Hugging Face のキャッシュ（`HF_HOME`）に 3.9 GB を取得する。容量の少ないディスクを避けるなら、明示したパスを渡す |
| mp4 の読み書きに失敗する | mediapy は `ffmpeg` を使う。`which ffmpeg` で確認する |

---

### 付録: 参考情報

- **主要なリポジトリ:** fork `yuki-inaho/cowtracker`（`blackwell`）、upstream `facebookresearch/cowtracker`、
  submodule `facebookresearch/vggt`・`DepthAnything/Depth-Anything-V2`。評価手順の参照元は `aharley/alltracker` の `test_dense_on_sparse.py`。
- **代表的なコマンド:** `uv run cow-track --help`、`uv run cow-rerun --help`、`uv run cow-video --help`、
  `uv run cow-eval-tapvid --help`。
- **データの取得:** `https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip`、`.../tapvid_rgb_stacking.zip`
  （RoboTAP は `.../robotap/robotap.zip`、13.5 GB）。
- **連絡先:** TBD（GitHub 上の fork の所有者を確認）。

### 更新履歴

- 2026-09-27: 初版。uv 環境、toolkit、Blackwell 対応、TAP-Vid の再現結果に合わせて作成。
- 2026-09-27: `cow-video`（AllTracker 形式の密なトラック動画）と `cow-track --query-frame` を追記。
