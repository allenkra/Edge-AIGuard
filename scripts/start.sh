#!/usr/bin/env bash
# Edge-AIGuard 启动: ollama daemon + warmup + pipeline
#
# Usage:
#   export CORE2_NOISE_PSK="..."          # 必需 (--core2*)
#   scripts/start.sh                      # 默认 --core2 --core2-audio
#   scripts/start.sh --text --no-audio    # 文本模式
set -euo pipefail
cd "$(dirname "$0")/.."

LLM_MODEL="${LLM_MODEL:-qwen2.5:1.5b}"

# ollama daemon (没起就起)
curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null \
  || { nohup ollama serve > /tmp/ollama.log 2>&1 & sleep 3; }

# warmup: 用 prompts.build_system_prompt() 真正吃一遍生产 prompt → cache hot
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

# 跑 pipeline (CORE2_NOISE_PSK 必须由 caller export)
source ~/ollama/bin/activate
exec python pipeline_pipecat.py "${@:---core2 --core2-audio}"
