#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PAPER="$ROOT/docs/paper_ii"
BUILD="$ROOT/tmp/pdfs/paper-ii"
PY=${PY:-python3}
TECTONIC=${TECTONIC:-$(command -v tectonic || printf '%s' "$HOME/.local/bin/tectonic")}

PYTHONDONTWRITEBYTECODE=1 "$PY" "$ROOT/scripts/paper_ii/revision_metrics.py" --figures
PYTHONDONTWRITEBYTECODE=1 "$PY" "$ROOT/scripts/paper_ii/make_figures.py"
mkdir -p "$BUILD"
for lang in en uk; do
  mkdir -p "$BUILD/$lang"
  (cd "$PAPER/tex/$lang" && "$TECTONIC" --keep-logs --outdir "$BUILD/$lang" main.tex)
  cp "$BUILD/$lang/main.pdf" "$PAPER/article-$lang.pdf"
  echo "Built $PAPER/article-$lang.pdf"
done
