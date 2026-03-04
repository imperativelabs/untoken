#!/usr/bin/env bash
set -euo pipefail
source ~/untoken-env/bin/activate

sudo mkdir -p /data/untoken
sudo chown "$USER" /data/untoken

cd ~/untoken
python -m data.download \
    --datasets meetingbank cnn_dailymail gsm8k \
    --output-dir /data/untoken

echo "Data download complete. Contents:"
ls /data/untoken/
