# RTX 2070 ローカルセットアップ

`origin/blackwell` の `73cb7ee` から作成した `rtx2070` ブランチ用の手順です。
この環境では既存の lock をそのまま使います。torch 2.7.0+cu128 は sm_75 を含み、
xformers 0.0.30 の cutlass 経路が RTX 2070 の fp16 推論に使われます。
Python 3.12、torchvision 0.22.0+cu128、timm 1.0.25、rerun-sdk 0.23.1 です。

## セットアップ

```bash
rtk git submodule update --init --recursive
rtk uv sync --locked
```

環境はこのリポジトリの `.venv` に作成します。他のリポジトリの環境は使いません。
動画の読み書きにはシステムの ffmpeg も必要です。

チェックポイントは `facebook/cowtracker` の revision
`860b85cfb3a63b91780535689c262b3e595b6082` に固定して取得します。
`checkpoints/cowtracker_model.pth`（3,898,890,821 bytes）に配置します。
このファイルにモデル全体の重みが含まれ、VGGT / Depth Anything の別チェックポイントは不要です。

```bash
rtk proxy mkdir -p checkpoints
rtk proxy curl -fL --retry 3 -o checkpoints/cowtracker_model.pth.part \
  https://huggingface.co/facebook/cowtracker/resolve/860b85cfb3a63b91780535689c262b3e595b6082/cowtracker_model.pth
rtk proxy sha256sum checkpoints/cowtracker_model.pth.part
# 下記の SHA-256 と一致したら移動
rtk proxy mv checkpoints/cowtracker_model.pth.part checkpoints/cowtracker_model.pth
```

SHA-256: `5b46c6dac1d9f9f5944371176f5aedec13522db4ea86d998f3445b0fd37ab784`。
コードの FAIR Noncommercial Research License と重みの CC-BY-NC-4.0 は維持されます。

## GPU を共有する場合の小さい動作確認

RGB-D データの場合も、CoWTracker へ渡すのは `scenes/<scene>/rgb` の画像列です。
深度、カメラ行列、学習用 split は、この RGB 追跡の動作確認では使いません。
入力の元サイズが 112 の倍数でなければ、`--size` を明示します。
以下の `<RGB画像フォルダ>` を実際のフォルダに置き換えてください。

```bash
rtk uv run --locked cow-track --source <RGB画像フォルダ> \
  --checkpoint checkpoints/cowtracker_model.pth --device cuda:0 --dtype fp16 \
  --size 224 336 --max-frames 8 --mode full --vram-limit-gb 4 \
  --out outputs/rtx2070/full --rrd

rtk uv run --locked cow-track --source <RGB画像フォルダ> \
  --checkpoint checkpoints/cowtracker_model.pth --device cuda:0 --dtype fp16 \
  --size 224 336 --max-frames 24 --mode auto --window-len 4 --vram-limit-gb 4 \
  --out outputs/rtx2070/windowed --rrd
```

`--vram-limit-gb 4` は PyTorch allocator の上限（GiB）です。CUDA context や他プロセスの使用量は別です。
空きが上限を下回ると起動時にエラーになります。上限だけでは GPU の計算使用率は制限されません。
フレーム数や窓を増やすと出力バッファも増えるため、長い動画が同じ上限で動く保証はありません。
既定の窓 100 と bf16 は、この RTX 2070 向け手順では使いません。

出力は `tracks.npz`、`tracks.mp4`、`tracks.rrd`、`meta.json` です。
任意フレームから前後へ追跡する場合は `--query-frame 3` などを追加します。
窓と解像度を変更した結果を、Blackwell の論文再現条件と同じ精度として扱わないでください。

## 検証

2026-09-27、RTX 2070（sm_75、8 GiB）、他プロセスと GPU を共有した状態で確認しました。
2 系列の実 RGB 画像（各 1,000 枚、288×384）を読み取り検査し、それぞれの先頭部分を使いました。
推論条件は 224×336、fp16、cutlass、上限 4 GiB です。

| 確認 | 入力と条件 | 推論時間 | peak allocated |
| :--- | :--- | ---: | ---: |
| 通常推論・前後追跡 | 系列 0、8 フレーム、query 3、5 + 4 フレームの 2 回 | 2.60 秒 | 2.50 GiB |
| 窓分割推論 | 系列 1、24 フレーム、query 0、窓 4、6 窓 | 10.65 秒 | 2.98 GiB |

時間は CLI が計測した推論部分で、重みロードや出力の保存は含みません。共有 GPU での参考値です。
最初の行の再現には、上記 full コマンドへ `--query-frame 3` を追加します。
両方ともチェックポイントのロードは `All keys matched successfully` でした。
出力 shape と全要素の有限性、vis/conf の [0,1] 範囲、query フレームの恒等座標（最大誤差 0 px）、
MP4 の全フレームのデコード、Rerun の全フレームの読み戻しを確認しました。
初期化ログには継承された `Flash Attention 3 enabled` という文言が出ますが、
実際に選択された演算は両方とも `meta.json` に記録された `cutlassF-pt` です。

これは実データでの動作確認です。全 2,000 フレームの推論や、正解トラックとの精度評価は行っていません。
CPU テストは **78 passed / 6 deselected**、toolkit と tests の Ruff check / format check、`uv build` も成功しました。

既存の `tests/test_gpu_e2e.py` は上限 30 GiB と bf16 を含む Blackwell 向けのため、
RTX 2070 ではそのまま一括実行せず、上記の明示した条件で検証します。
ローカルの入力データ、重み、生成物、ログは `temp/`、`checkpoints/`、`outputs/` に置き、commit しません。

## 長辺500 px以上・上限6 GiB・窓分割なし

元画像の幅384×高さ288（4:3）に近く、フレーム数を確保する条件として、
幅560×高さ448（5:4）を使います。両辺とも112の倍数で、比率は元より6.25%横に狭くなります。
完全に4:3を維持すると最小は幅896×高さ672になり、5フレームで5.19 GiBを使用しました。
元画像を拡大するので、入力自体の細部が増えるわけではありません。

```bash
rtk uv run --locked cow-track --source <RGB画像フォルダ> \
  --checkpoint checkpoints/cowtracker_model.pth --device cuda:0 --dtype fp16 \
  --size 448 560 --max-frames 14 --mode full --vram-limit-gb 6 \
  --out outputs/rtx2070/full_560x448_14 --rrd
```

実データの先頭14フレームを単一パスで処理し、peak allocated は **5.755 GiB**、
推論は **9.47秒**でした。query 0、fp16、cutlass、窓分割なしです。
14枚すべての動画・Rerun読み戻し、追跡座標の有限性、vis/confの範囲、先頭フレームの恒等座標を確認しました。
上限6 GiBはPyTorch allocatorに対する値で、CUDA contextや画面表示は別です。

同条件の15フレームは6 GiB allocator上限でOOMになりました（38 MiB追加確保に失敗）。
この環境・解像度・fp16・現行実装で、14フレームが成功し、直上の15フレームが失敗したことを確認しています。
