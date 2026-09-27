# 推論の精度（fp16 / bf16 / fp32）と RTX 2070 で動かすためのメモリ調査

> この文書は Blackwell 上での調査記録です。`rtx2070` ブランチでの実機セットアップと
> 上限 4 GiB / 6 GiB での動作確認結果は [rtx2070_setup.md](rtx2070_setup.md) を参照してください。

2026-09-27 に RTX 5090（32 GB）で測りました。RTX 2070（Turing、sm_75、8 GB）の実機は手元に無いため、
RTX 5090 の上で、VRAM の上限と attention の kernel を RTX 2070 と同じ条件にして模擬しています。
速度は模擬できないので、実機での時間は未測定です。

## 結論

- **fp16 推論はできます。既定も fp16 です（upstream の demo と同じ）。** fp32 と比べた track の差は、
  見えている画素で平均 0.03 px、全画素の中央値で 0.03 px、可視判定の一致率は 99.99% です。
  TAP-Vid DAVIS（30 本、窓 64）では、fp16 と bf16 の差は 0.02 以下でした。fp32 の DAVIS 評価は途中で打ち切ったため、
  数値はありません（E2）。fp32 は約 5 倍遅く、メモリも 1.5 倍使うので、使う理由はありません。
- bf16 は fp16 より誤差が約 5 倍大きく（見えている画素で平均 0.19 px）、しかも Turing では使えません。
  RTX 2070 では fp16 を使います。
- RTX 2070 で動かすには、次の 3 つが必要です。
  1. **attention の kernel**: xformers の FlashAttention2 は sm_80 以上なので使えません。
     このブランチでは、compute capability が 8.0 未満なら自動で cutlass kernel を選びます（精度は FA2 と同等）。
  2. **VRAM**: 重みが 1.82 GB に、1 フレームあたり 0.085〜0.211 GB（解像度による）が加わります。
     8 GB に収めるには、1 回の推論のフレーム数を減らす（`--window-len` を短くする）か、解像度を下げます。
  3. **ホストの RAM**: checkpoint（fp32、3.9 GB）の読み込みで、ピークが約 8 GB になります。16 GB 以上を推奨します。
- RTX 2070 相当の条件（上限 7.0 GB、fp16、cutlass）での DAVIS の評価（E3）は、実行前に打ち切りました。
  下の推奨設定は、メモリのモデル式から計算した値で、実行して確かめてはいません。
- CPU 推論は選択肢に含めません（このブランチの CPU 経路は、重みを使わないテストのためだけのものです）。

## dtype と attention の kernel

video transformer の空間 attention（xformers）の kernel は、GPU の世代と dtype で、次のように自動で選びます。
VGGT の aggregator と temporal attention は、PyTorch の SDPA です。

| GPU（compute capability） | fp16 / bf16 | fp32 | 備考 |
| :--- | :--- | :--- | :--- |
| Hopper（9.x） | FlashAttention3 | cutlass | upstream と同じ |
| Ampere / Ada / Blackwell（8.0〜、10.x、12.x） | FlashAttention2 | cutlass | FA3 は sm_90 専用で、Blackwell ではプロセスが落ちる |
| Turing（7.5、RTX 20xx）以前 | cutlass | cutlass | FA2 は sm_80 以上。bf16 は Turing では非対応 |

- upstream のままだと、fp32 は CUDA でも `memory_efficient_attention does not support inputs` で失敗し、
  Turing でも FA2 が選ばれて失敗します。このブランチで直しました。
- 使った kernel は、`meta.json` / `summary.json` の `attention` に記録されます（例: `fa2F@2.5.7-pt`、`cutlassF-pt`）。
- `--dtype auto|fp16|bf16|fp32`（auto は fp16）で切り替えられます。

## 精度（E1: 密なトラックの差）

bmx の 48 フレーム、336×560、単一パスで、fp32 を基準にした差です（フレーム 1〜47 の全画素）。

| 条件 | kernel | EPE 平均 | 中央値 | p99 | 可視画素の EPE 平均 | vis / conf の MAE | 可視判定の一致（@0.6） | 時間 | peak VRAM |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| fp16（既定） | FA2 | 0.099 px | 0.032 | 0.94 | 0.026 | 0.0001 / 0.0005 | 99.99% | 3.0 s | 11.9 GB |
| fp16（Turing 相当） | cutlass | 0.102 px | 0.032 | 0.94 | 0.026 | 0.0002 / 0.0006 | 99.99% | 4.1 s | 11.9 GB |
| bf16 | FA2 | 0.499 px | 0.223 | 3.38 | 0.193 | 0.0009 / 0.0041 | 99.91% | 3.0 s | 11.9 GB |
| fp32（基準） | cutlass | 0 | 0 | 0 | 0 | 0 | 100% | 14.2 s | 17.9 GB |

- 「可視画素」は、fp32 で vis×conf ≥ 0.6 の画素です。最大値（fp16 で 254 px）は、ごく一部の画素が別の対応点に
  飛んだもので、p99 は 1 px 未満です。
- 時間は RTX 5090 での値です。Turing 相当の kernel は、この GPU では FA2 より約 35% 遅くなります。

## 精度（E2: TAP-Vid DAVIS、窓をそろえた dtype の比較）

fp32 が 30 GB に収まるよう、窓はどれも 64 にそろえています（336×560、`--mode auto`、可視の閾値 0.6）。

| dtype | 窓 | AJ | δavg | OA | 時間（30 本） | peak VRAM |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16（FA2） | 64 | 66.23 | 78.40 | 91.36 | 364 s | 16.1 GB |
| bf16（FA2） | 64 | 66.22 | 78.40 | 91.34 | 360 s | 16.1 GB |
| fp32（cutlass） | 64 | 未完了（打ち切り） | | | | |
| fp16（参考: 窓 100、[uv_tools.md](uv_tools.md) の評価） | 100 | 66.22 | 78.43 | 91.29 | 369 s | 23.6 GB |

- 同じ窓の fp16 と bf16 の差は、AJ 0.01、δavg 0.00、OA 0.02 で、評価の上では区別できません。
- 窓を 100 から 64 に縮めても、DAVIS の差は 0.07 以下でした。DAVIS の動画は 34〜104 フレームで、
  窓 64 を超えるのは一部の動画だけです。長い動画での窓の影響は測っていません。

## メモリのモデル（E0）

fp16、単一パス、Turing 相当の kernel で測ると、peak VRAM（allocated）はフレーム数 T に対して完全に直線でした
（8・16・24 フレームで当てはめた式が、40 フレームの実測と一致）。この GPU 本来の kernel でも、値は同じです。

**peak ≈ 1.82 GB + b × T**

| 解像度 | b（GB/フレーム） | 16 フレーム | 24 フレーム | 40 フレーム | 8 GB に収まるフレーム数の目安 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 224×336 | 0.085 | 3.2 GB | 3.9 GB | 5.2 GB | 約 55 |
| 336×448 | 0.169 | 4.5 GB | 5.9 GB | 8.6 GB | 約 27 |
| 336×560 | 0.211 | 5.2 GB | 6.9 GB | 10.3 GB | 約 22 |

- 目安は、上限 7.0 GB から、出力バッファなどの余裕 0.5 GB を引いて計算しています。
- 実際に GPU が確保する量（reserved）は、peak の約 1.1 倍です。
- 重みは fp16 で 1.82 GB（974.5M パラメータ）です。ロードは CPU で fp16 に変換してから GPU に移すので、
  fp32 の 3.6 GB が GPU に載ることはありません。
- 長い動画は windowed 推論（先頭 1 + メモリ 10 + 窓 W フレーム）で処理します。1 回に処理するフレーム数は 11 + W です。
  加えて、動画全体の出力バッファとして、約 40 B/画素/フレーム が GPU に載ります
  （336×448 で 100 フレームなら約 0.6 GB）。

## RTX 2070 の模擬（E3: DAVIS 30 本）

上限 7.0 GB（物理 8 GB から、表示と CUDA context の分を引いた保守的な値）、fp16、
xformers は cutlass、SDPA の flash kernel は停止、という条件です。窓 W は、上のモデル式で 6.5 GB 以下になる
最大の 4 の倍数にしました。

実行前に打ち切ったため、結果はありません。模擬に使う予定だった条件は次のとおりです（窓は、メモリのモデル式から計算）。

| 解像度 | 窓 W | 1 回の推論のフレーム数 | 推定 peak（出力バッファ込み） |
| :--- | ---: | ---: | ---: |
| 224×336 | 40 | 51 | 6.45 GB |
| 336×448 | 12 | 23 | 6.29 GB |
| 336×560 | 4 | 15 | 5.72 GB |

模擬の方法は、次の 2 点を実行時に差し替えるものです。
- xformers の attention を cutlass にする（このブランチのコードが capability 7.5 で選ぶものと同じ）。
- `torch.backends.cuda.enable_flash_sdp(False)` にする（Turing には PyTorch の flash kernel が無いため）。

メモリの使い方は、この GPU 本来の kernel と同じでした（E0）。

## RTX 2070 での推奨設定

計算上の値です。実機でも、模擬でも確かめていません。

- **dtype は fp16（既定）。** bf16 は Turing では使えず、fp32 はメモリが 1.5 倍になります。
- **解像度と窓:** 224×336 なら `--window-len 40`、336×448 なら `--window-len 12` が 7 GB に収まる計算です。
  336×560 は窓 4 になり、ほぼメモリフレームだけの推論になるので勧めません。
  窓を短くすると、長い動画での精度が落ちる可能性があります（未測定）。
- 上限は `--vram-limit-gb 7` を付けます（空きが 7 GB 未満なら、起動時に止まります）。

```bash
uv run cow-track --source <動画か画像フォルダ> --size 224 336 --window-len 40 --dtype fp16 \
  --vram-limit-gb 7 --out outputs/<名前> --checkpoint checkpoints/cowtracker_model.pth
```

## ホストの RAM とロード

- checkpoint は fp32 の 3.9 GB で、ロードのピークでホストの RSS が約 8 GB になります。16 GB 以上の RAM を推奨します。
- Hugging Face から取得する場合は、キャッシュ（`HF_HOME`）に 3.9 GB 必要です。

## 未検証事項

- RTX 2070 の実機での速度と、表示などに取られる VRAM の実際の量。
- Turing での xformers の cutlass kernel の実行（`_C.so` に sm_75 の SASS が入っていることは確認済み）。
- `--window-len` 以外の軽量化（メモリフレーム数 10 の変更、解像度の組み合わせなど）の精度への影響。
