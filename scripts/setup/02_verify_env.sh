#!/usr/bin/env bash
set -euo pipefail
source ~/untoken-env/bin/activate

python - <<'EOF'
import torch

assert torch.cuda.is_available(), "CUDA not available"
n = torch.cuda.device_count()
assert n > 0, f"No GPUs found"

for i in range(n):
    name = torch.cuda.get_device_name(i)
    print(f"  GPU {i}: {name}")

assert torch.cuda.is_bf16_supported(), "BF16 not supported"
assert hasattr(torch, "compile"), "torch.compile missing"

print(f"PASS — {n}x {torch.cuda.get_device_name(0)}")
print(f"       CUDA {torch.version.cuda}, PyTorch {torch.__version__}")
EOF
