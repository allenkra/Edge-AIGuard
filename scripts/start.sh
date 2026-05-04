#!/usr/bin/env bash
# Edge-AIGuard launcher: ollama daemon + LLM warmup + pipeline.
#
# Usage:
#   scripts/start.sh                      # default: --core2 --core2-audio
#   scripts/start.sh --text --no-audio    # text-only mode
set -euo pipefail
cd "$(dirname "$0")/.."

LLM_MODEL="${LLM_MODEL:-qwen2.5:1.5b}"

# 1. Start ollama daemon if not running.
if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null; then
    echo "[start] launching ollama..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi

# 2. Auto-load CORE2_NOISE_PSK from esphome/secrets.yaml if unset.
if [[ -z "${CORE2_NOISE_PSK:-}" ]]; then
    if [[ -f esphome/secrets.yaml ]]; then
        CORE2_NOISE_PSK=$(grep '^api_password:' esphome/secrets.yaml | cut -d'"' -f2)
        export CORE2_NOISE_PSK
        echo "[start] CORE2_NOISE_PSK loaded from esphome/secrets.yaml"
    else
        echo "[start] WARNING: CORE2_NOISE_PSK not set and esphome/secrets.yaml not found" >&2
    fi
fi
export CORE2_HOST="${CORE2_HOST:-edge-aiguard-core2.local}"

# 3. Warmup: run the production system prompt once to populate the prompt cache.
echo "[start] warming $LLM_MODEL with production system prompt..."
python3 -c "
import sys, json, urllib.request
sys.path.insert(0, '.')
from prompts import build_system_prompt
sp = build_system_prompt({'hr':72,'br':16,'presence':True,'category':'normal'})
body = json.dumps({'model':'$LLM_MODEL','system':sp,'prompt':'warmup','stream':False,
                   'keep_alive':'30m','options':{'num_predict':1}}).encode()
urllib.request.urlopen(urllib.request.Request(
  'http://localhost:11434/api/generate', data=body,
  headers={'Content-Type':'application/json'}), timeout=120).read()
print(f'  cached {len(sp)} chars')
"

# 4. Launch pipeline.
source ~/ollama/bin/activate
if [ $# -eq 0 ]; then
    set -- --core2 --core2-audio
fi
exec python pipeline_pipecat.py "$@"
