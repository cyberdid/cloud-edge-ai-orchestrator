#!/usr/bin/env bash
# Prepare a fresh Colab VM (run as root): code, pinned Python envs, Ollama + model,
# bucket credentials and restored run state.  Usage: bootstrap.sh <repo.tar.gz> <model-tag>
set -euo pipefail
TARBALL="$1"
MODEL="$2"
BUCKET=gs://oe-colab-scale-0903767759
export UV_PYTHON_INSTALL_DIR=/opt/uv-python   # readable by the candidate user, unlike /root
export CLOUDSDK_CONFIG=/root/.oe/gcloud

chmod 700 /root
mkdir -p /root/oe /root/oe-runs /root/.oe
chmod 700 /root/.oe
tar -xzf "$TARBALL" -C /root/oe
cd /root/oe

command -v uv >/dev/null || pip install -q uv
uv sync --locked --extra evolve --extra research -q

# Interpreter for candidate code: the Docker sandbox's pinned runtime, world-readable.
if [ ! -x /opt/oe-cand/bin/python ]; then
  uv venv /opt/oe-cand --python 3.12 -q
  uv pip install --python /opt/oe-cand/bin/python -q -r antigravity_hybrid_orchestrator/runtime/requirements.lock
fi
chmod -R a+rX /opt/oe-cand /opt/uv-python

if ! command -v ollama >/dev/null || ! ollama --version 2>/dev/null | grep -q 0.40.13; then
  command -v zstd >/dev/null || { apt-get -qq update && apt-get -qq install -y zstd >/dev/null; }
  curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.40.13 sh >/dev/null 2>&1
fi
if ! curl -sf localhost:11434/api/version >/dev/null; then
  OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=2 OLLAMA_KEEP_ALIVE=-1 \
    nohup ollama serve > /root/.oe/ollama.log 2>&1 &
  for _ in $(seq 60); do curl -sf localhost:11434/api/version >/dev/null && break; sleep 1; done
fi
ollama pull "$MODEL" > /root/.oe/pull.log 2>&1
DIGEST=$(ollama list | awk -v m="$MODEL" '$1==m {print $2}')

gcloud auth activate-service-account --key-file /root/.oe/sa-key.json --quiet >/dev/null 2>&1
if gcloud storage ls "$BUCKET/runs/" >/dev/null 2>&1; then
  gcloud storage rsync -r "$BUCKET/runs" /root/oe-runs --quiet >/dev/null 2>&1
fi

echo "BOOTSTRAP OK ollama=$(curl -s localhost:11434/api/version) model=$MODEL digest=$DIGEST unshare_net=$(unshare -n true 2>/dev/null && echo yes || echo no)"
