#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PAPER="$ROOT/docs/paper_ii"
BUILD="$ROOT/tmp/pdfs/paper-ii"
PY=${PY:-python3}
TECTONIC=${TECTONIC:-$(command -v tectonic || printf '%s' "$HOME/.local/bin/tectonic")}
mapfile -t CHECKSUM_FILES < <(awk 'NF == 2 {print $2}' "$PAPER/SHA256SUMS")

PYTHONDONTWRITEBYTECODE=1 "$PY" "$ROOT/scripts/paper_ii/revision_metrics.py" --figures
PYTHONDONTWRITEBYTECODE=1 "$PY" "$ROOT/scripts/paper_ii/make_figures.py"
mkdir -p "$BUILD"
for lang in en uk; do
  mkdir -p "$BUILD/$lang"
  (cd "$PAPER/tex/$lang" && "$TECTONIC" --keep-logs --outdir "$BUILD/$lang" main.tex)
  cp "$BUILD/$lang/main.pdf" "$PAPER/article-$lang.pdf"
  echo "Built $PAPER/article-$lang.pdf"
done

# Tectonic and Matplotlib may embed build timestamps. Refresh the manifest from
# its existing file list so it always verifies the exact generated package.
(cd "$ROOT" && sha256sum "${CHECKSUM_FILES[@]}" > "$BUILD/SHA256SUMS")
mv "$BUILD/SHA256SUMS" "$PAPER/SHA256SUMS"
(cd "$ROOT" && sha256sum -c "$PAPER/SHA256SUMS")
