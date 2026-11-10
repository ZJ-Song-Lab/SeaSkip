# SeaSkip training chain (PC): gate cls -> SSDD det -> HRSID det
# torch 2.14.0+cu126 already installed. Usage: powershell -ExecutionPolicy Bypass -File train_chain.ps1
# Progress: runs\train_chain.log ; per-run details: runs\*_train.log
# NOTE: yolo is invoked via cmd /c with shell-level redirection so its stderr never
#       becomes PowerShell ErrorRecords (EAP=Stop killed the chain once because of that).
$root = $PSScriptRoot
$log  = Join-Path $root 'runs\train_chain.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
function Log($m) { "{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m | Tee-Object -FilePath $log -Append }

Log "verifying CUDA..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, torch.version.cuda)"
if ($LASTEXITCODE -ne 0) { Log "[FATAL] torch CUDA verify failed, chain aborted"; exit 1 }

Set-Location $root
Log "[1/4] gate 3-way classifier (yolo11n-cls, 640, 20ep, batch 16)..."
cmd /c "yolo classify train model=yolo11n-cls.pt data=$root\gate\cls imgsz=640 epochs=20 batch=16 seed=42 workers=2 cache=False project=$root\runs\gate name=v1 > $root\runs\gate_v1_train.log 2>&1"
if ($LASTEXITCODE -ne 0) { Log "[FATAL] gate train failed(exit=$LASTEXITCODE), see runs\gate_v1_train.log"; exit 1 }
Log "gate done -> runs\gate\v1"

Log "[2/4] SSDD detector baseline (yolo11n, 640, 150ep, batch 8)..."
cmd /c "yolo detect train model=yolo11n.pt data=$root\processed\ssdd\ssdd.yaml imgsz=640 epochs=150 batch=8 seed=42 workers=2 cache=False project=$root\runs\det name=ssdd_v1 > $root\runs\det_ssdd_v1_train.log 2>&1"
if ($LASTEXITCODE -ne 0) { Log "[FATAL] SSDD train failed(exit=$LASTEXITCODE), see runs\det_ssdd_v1_train.log"; exit 1 }
Log "SSDD done -> runs\det\ssdd_v1"

Log "[3/4] HRSID detector baseline (yolo11n, 640, 150ep, batch 8)..."
cmd /c "yolo detect train model=yolo11n.pt data=$root\processed\hrsid\hrsid.yaml imgsz=640 epochs=150 batch=8 seed=42 workers=2 cache=False project=$root\runs\det name=hrsid_v1 > $root\runs\det_hrsid_v1_train.log 2>&1"
if ($LASTEXITCODE -ne 0) { Log "[FATAL] HRSID train failed(exit=$LASTEXITCODE), see runs\det_hrsid_v1_train.log"; exit 1 }

Log "[4/4] ALL TRAININGS DONE. Next: export ONNX -> ATC to OM -> deploy on Atlas 200I (paper/03)"