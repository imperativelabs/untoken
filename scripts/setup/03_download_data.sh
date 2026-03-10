#!/usr/bin/env bash
set -euo pipefail
source ~/untoken-env/bin/activate

sudo mkdir -p /data/untoken
sudo chown "$USER" /data/untoken

cd ~/untoken
python -m data.download \
    --datasets meetingbank cnn_dailymail gsm8k \
                booksum xsum multi_news samsum dialogsum \
                scientific_papers billsum big_patent wikihow reddit_tifu \
    --output-dir /data/untoken

echo "Data download complete. Contents:"
ls /data/untoken/
