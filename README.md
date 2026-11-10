# SeaSkip

SeaSkip is a cost-model-driven, out-of-graph early-exit system for SAR ship detection on static-graph edge NPUs. A 187,587-parameter three-state TinyGate runs as a separate static subgraph; a C++ AscendCL orchestrator skips the INT8 YOLO11n detector on patches predicted to be empty.

The implementation targets the Huawei Atlas 200I DK A2 (Ascend 310B4). The gate and detector decisions are made outside the compiled graphs, so the approach does not require dynamic control flow inside an OM model.

## Repository layout

```text
data/
  common.py                    Shared dataset and patch utilities
  voc2yolo.py                  VOC SSDD conversion
  ssdd2yolo.py                 Official SSDD conversion
  coco2yolo.py                 HRSID conversion
  mine_gate_patches.py         Empty/simple/complex gate-label mining
  build_patch_stream.py        Full-scene patch-stream construction
  train_chain.ps1              Detector and legacy v1 gate training chain
  train_tiny_gate.py           TinyGate v2/v3/v4 training and ONNX export
  train_tiny_gate_bin.py       Binary-gate ablation training
  eval_board_m3m6.py           Board latency and gate-quality analysis
  eval_skip_ap_roc.py          Detection AP and safety-threshold analysis
  analyze_c10_confidence.py    Five-run paired board analysis
  eval_binary_ablation.py      Three-class versus binary gate analysis
  eval_cfar_baseline.py        PC CA-CFAR baseline analysis
  analyze_v4_failures.py       False-skip error analysis
  pc_ref_predict.py            PC reference detector inference
  board/
    seaskip_eval.cpp           Final v4 AscendCL evaluator
    cfar_eval.cpp              Board CA-CFAR evaluator
    stb_image.h                Third-party JPEG decoder
    stb_impl.cpp               STB implementation for CA-CFAR
    run_c10.sh                 Five-run board protocol
  runs/
    gate/v4/summary.json
    gate/v4bin/summary.json
    board_eval/results_snapshot.json
    board_eval/v4/*.json
    board_eval/v4/c10_summary.csv
    board_eval/cfar/cfar_hrsid_test.json
    board_eval/cfar/results_cfar_board.csv
```

Datasets, derived patch images, model weights, OM models, and per-patch C10 timing logs are not included. The published JSON and CSV artifacts contain the final aggregate and per-run results needed to check every number below.

## Main results

Unless stated otherwise, results use the 7,848-patch HRSID full-scene test stream. Its ground-truth empty rate is 17.6%; the v4 gate skips 1,508 patches (19.2%) under argmax and 1,376 (17.5%) at `p_empty >= 0.6`.

### Final v4 system

| Metric | Result |
|---|---:|
| TinyGate parameters | 187,587 |
| Gate validation accuracy | 84.29% |
| Board gate-input construction | 1.79 ms/patch |
| Board gate NPU inference | 0.61 ms/patch |
| Total gate-path overhead | 2.40 ms/patch |
| Break-even skip rate | 16.2% |
| Full-stream patch AP@50 | 0.7522 |
| Argmax-skip patch AP@50 | 0.7408 |
| AP@50 change | -0.0114 |
| False-skipped patches, argmax | 230 |
| False-skipped patches, `tau=0.6` | 146 |
| Saving in one clean board run | +1.32% |
| Replay-estimated saving at 90% empty | +30.03% |
| Replay-estimated saving at 95% empty | +32.09% |

The sparsity values are measured-timing replay results, not additional board runs over synthetic scenes.

### C10: five paired board runs

Each pair uses one full run and one skip run. Values below are derived from `c10_confidence.json`; the table is also available as `c10_summary.csv`.

| Policy | Saving, mean +/- sample SD | Break-even skip rate, mean +/- sample SD |
|---|---:|---:|
| Argmax | -0.1% +/- 1.3% | 19.0% +/- 3.4% |
| `tau=0.6` | +0.09% +/- 1.6% | 19.0% +/- 3.4% |

The measured stream is at the break-even point. The single-run +1.32% result and the five-run distribution are both reported; they are not interchangeable.

### C12: tuned CA-CFAR prescreener

| Method | Policy | False-skipped patches | Board timing result |
|---|---|---:|---:|
| TinyGate v4 | Argmax | 230 | +0.12% five-run mean saving |
| TinyGate v4 | `tau=0.6` | 146 | +0.09% five-run mean saving |
| CA-CFAR | Argmax-equivalent (`alpha=6`, `min_pix=1`) | 49 | 17.40% regression |
| CA-CFAR | Safety threshold (`tau=0.6`) | 92 | 3.33% regression |

CA-CFAR costs 7.87 ms/patch on the board, versus 0.61 ms NPU inference for TinyGate. The complete PC tuning grid, PC timing, and false-skip lists are in `cfar/cfar_hrsid_test.json`; board rows are in `cfar/results_cfar_board.csv`.

### B7: binary-gate ablation

| Gate | Policy | Skip rate | False-skipped patches | Lost GT ships | Ship recall |
|---|---|---:|---:|---:|---:|
| Three-class v4 | Argmax | 19.22% | 230 | 521 | 86.48% |
| Binary v4bin | Argmax | 15.40% | 150 | 281 | 90.77% |
| Three-class v4 | `tau=0.6` | 17.53% | 146 | 230 | 89.07% |
| Binary v4bin | `tau=0.6` | 12.50% | 125 | 196 | 91.28% |

The binary gate is safer but skips less work. See `v4/binary_ablation.json` and `gate/v4bin/summary.json`.

## Result artifacts

| Artifact | Contents |
|---|---|
| `results_snapshot.json` | v1-v4 iteration history and final v4 results |
| `v4/m3m6_hrsid_test.json` | Gate confusion, false skips, and board stage timings |
| `v4/skip_ap_roc_hrsid_test.json` | AP, safety ROC, and sparsity replay |
| `v4/c10_confidence.json` | Five paired runs, per-pair values, mean/SD/min/max |
| `v4/c10_summary.csv` | Flat ten-row version of the paired C10 results |
| `v4/binary_ablation.json` | Three-class/binary parity and ablation metrics |
| `v4/failure_analysis.json` | D13 false-skip analysis and per-patch failure list |
| `cfar/cfar_hrsid_test.json` | Complete PC CA-CFAR tuning and test report |
| `cfar/results_cfar_board.csv` | Per-patch board CA-CFAR decisions and timings |

## Reproduction

### 1. Python environment

Python 3.12 and a CUDA-capable PyTorch installation are recommended for training. The recorded detector environment used Ultralytics 8.4.189 and ONNX 1.23.2.

```bash
python -m pip install -r requirements.txt
```

### 2. Prepare datasets

Download SSDD and HRSID from their official sources and place them in local directories. Choose `voc2yolo.py` or `ssdd2yolo.py` according to the SSDD layout; do not run both into the same output tree.

```bash
python data/ssdd2yolo.py --ssdd-root <OFFICIAL_SSDD> --out data/processed/ssdd
python data/coco2yolo.py --coco-root <HRSID> --out data/processed/hrsid --panorama-dir <HRSID_PANORAMAS>
python data/mine_gate_patches.py --datasets ssdd hrsid --processed-root data/processed --out data/gate
python data/build_patch_stream.py --dataset hrsid --split test --processed-root data/processed --out data/board
```

### 3. Train and export

```powershell
powershell -ExecutionPolicy Bypass -File data/train_chain.ps1
python data/train_tiny_gate.py --v4
python data/train_tiny_gate_bin.py
```

`train_chain.ps1` trains the legacy v1 gate and YOLO11n detector baselines. The final system uses the v4 TinyGate from `train_tiny_gate.py`. Export the detector to INT8 OM and the gate to a fixed-shape raw-logits OM with the Ascend ATC toolchain; calibration data and trained weights are required for this step.

### 4. Build and run on the Atlas board

Install CANN/AscendCL and copy the contents of `data/board/` to the same board directory. The recorded board environment was Atlas 200I DK A2, Ascend310B4, CANN 23.0.rc3.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
g++ -O2 -std=c++17 seaskip_eval.cpp -o seaskip_eval \
  -I"$ASCEND_TOOLKIT_HOME/include" -L"$ASCEND_TOOLKIT_HOME/lib64" -lascendcl
./seaskip_eval <gate.om> <det.om> <patches_dir> <out.csv> <full|skip|gate>
```

For CA-CFAR:

```bash
g++ -O2 -std=c++17 cfar_eval.cpp stb_impl.cpp -o cfar_eval
./cfar_eval <patches_dir> <out.csv> <alpha> <min_pix>
```

To repeat C10, place `gate_tiny.om`, `det_int8.om`, `patches/hrsid_test`, `run_c10.sh`, and the compiled evaluator under `~/seaskip`, then run:

```bash
bash run_c10.sh
```

### 5. Recompute reports

Run these scripts from the repository root after restoring the required local datasets, weights, and board CSVs:

```bash
python data/eval_board_m3m6.py
python data/eval_skip_ap_roc.py
python data/analyze_c10_confidence.py
python data/eval_binary_ablation.py
python data/eval_cfar_baseline.py
python data/analyze_v4_failures.py
```

The C10 analyzer expects ten per-patch CSVs under `data/runs/board_eval/v4/c10/` using the names documented in its source. Those large logs are not bundled; `c10_confidence.json` preserves all aggregate and per-pair outputs.

## Third-party notice

`stb_image.h` is stb_image v2.30 by Sean Barrett and contributors, distributed under the public domain or MIT license at the user's option. Its original copyright and license text is retained in the header.