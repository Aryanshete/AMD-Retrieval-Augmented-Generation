#!/usr/bin/env bash
# Pre-submission checks, mirroring the grader's hard gates (Match 3 brief, "Check your
# submission before you send it").
#
#   ./scripts/check_submission.sh IMAGE[:TAG] /path/to/mc3-starter-kit
#
# Needs docker and an AMD GPU (/dev/kfd, /dev/dri). Exit code 0 = every gate passed.
set -uo pipefail

IMAGE="${1:?usage: $0 IMAGE[:TAG] /path/to/mc3-starter-kit}"
KIT="${2:?usage: $0 IMAGE[:TAG] /path/to/mc3-starter-kit}"
BASE_IMAGE="${BASE_IMAGE:-rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SIZE_LIMIT_GIB=60
STARTUP_LIMIT_S=600
NAME="sourcetrace-check-$$"
FAILED=0

pass() { printf '  [ ok ] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; FAILED=1; }
info() { printf '\n== %s\n' "$1"; }

TMP="$(mktemp -d)"
cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  chmod -R u+rw "$TMP" 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

command -v docker >/dev/null || { echo "docker not found"; exit 2; }
[ -d "$KIT/mc3-corpus" ] || { echo "no mc3-corpus in $KIT"; exit 2; }

info "1. Image size, uncompressed (limit ${SIZE_LIMIT_GIB} GiB)"
BYTES="$(docker image inspect "$IMAGE" --format '{{.Size}}')" || { echo "image not found: $IMAGE"; exit 2; }
GIB="$(python3 -c "print(round($BYTES/1024**3, 2))")"
if python3 -c "import sys; sys.exit(0 if $BYTES <= $SIZE_LIMIT_GIB*1024**3 else 1)"; then
  pass "${GIB} GiB"; else fail "${GIB} GiB is over the limit"; fi

info "2. Built on the mandated base image (layer identity, never squashed)"
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull -q "$BASE_IMAGE" >/dev/null 2>&1
if python3 - "$IMAGE" "$BASE_IMAGE" <<'PY'
import json, subprocess, sys
def layers(ref):
    return json.loads(subprocess.check_output(["docker", "image", "inspect", ref, "--format", "{{json .RootFS.Layers}}"]))
mine, base = layers(sys.argv[1]), layers(sys.argv[2])
sys.exit(0 if mine[: len(base)] == base else 1)
PY
then pass "base layers are the first layers of the image"; else fail "not built on $BASE_IMAGE"; fi

info "3. torch is still the ROCm build"
VER="$(docker run --rm --entrypoint python3 "$IMAGE" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
case "$VER" in *+rocm*) pass "torch $VER";; *) fail "torch is '$VER' (must contain +rocm)";; esac

info "4. Prepare a corpus with the four nasty cases"
python3 "$HERE/prepare_corpus.py" --src "$KIT/mc3-corpus" --dest "$TMP/corpus" | sed 's/^/  /'

info "5. Start with NO network and WITHOUT DAC_OVERRIDE (like the grader)"
docker run -d --name "$NAME" --network none --cap-drop DAC_OVERRIDE \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --security-opt seccomp=unconfined "$IMAGE" >/dev/null || { fail "container did not start"; exit 1; }
START="$(date +%s)"
docker exec "$NAME" mkdir -p /app/corpus /app/output
docker cp "$TMP/corpus/." "$NAME:/app/corpus" >/dev/null
# docker cp as root may not keep chmod 000 as the harness does; enforce it inside.
docker exec "$NAME" chmod 000 /app/corpus/vendor/internal_audit.txt 2>/dev/null || true

info "6. --index over the whole corpus (counts against the ${STARTUP_LIMIT_S}s startup budget)"
if timeout "$STARTUP_LIMIT_S" docker exec "$NAME" python3 /app/app.py --index /app/corpus; then
  ELAPSED=$(( $(date +%s) - START ))
  if [ "$ELAPSED" -lt "$STARTUP_LIMIT_S" ]; then pass "container start + model load + index = ${ELAPSED}s"
  else fail "startup took ${ELAPSED}s"; fi
else fail "--index failed or timed out"; docker logs --tail 40 "$NAME"; fi

info "7. Ten sample questions: answer AND exact citations AND < 30 s each"
if python3 "$HERE/evaluate.py" --container "$NAME" --questions "$KIT/sample-questions.json"; then
  pass "all sample questions scored"; else fail "some sample questions did not score"; fi

info "8. Container still running after all questions"
[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = "true" ] && pass "still up" || fail "container exited"

info "9. GPU memory (must stay between 1 and 48 GiB)"
if command -v amd-smi >/dev/null; then
  amd-smi metric --mem 2>/dev/null | grep -i -E 'used_vram|total_vram' | head -4 | sed 's/^/  /'
else echo "  amd-smi not found; check manually: watch -n1 'amd-smi metric --mem'"; fi

info "Result"
if [ "$FAILED" = 0 ]; then echo "  all checks passed"; else echo "  SOME CHECKS FAILED"; fi
echo "  Reminder: the graded corpus is different, larger and harder than this sample."
exit "$FAILED"
