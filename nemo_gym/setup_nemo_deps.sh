#!/bin/bash
# Build minimal nemo-gym deps dir: --no-deps install + iterative import probe.
# Container packages win for anything it already ships; we vendor only the gaps.
set -euo pipefail
DEPS=/iopsstor/scratch/cscs/rosmith/nemo_gym_pydeps
PINS=/iopsstor/scratch/cscs/rosmith/nemo_gym_pins
SRC=/iopsstor/scratch/cscs/rosmith/NemoGym-sglang
rm -rf "$DEPS" "$PINS"
PIPI='pip install -q --target '"$DEPS"' --no-deps --ignore-installed --upgrade'
$PIPI "$SRC"
pip install -q --target "$PINS" --no-deps --ignore-installed --upgrade "anthropic<=0.109.2"
$PIPI uv
pip install -q --target "$PINS" --no-deps --ignore-installed --upgrade "openai<=2.7.2"
test -f "$PINS/anthropic/__init__.py" || { echo "FATAL: anthropic did not land in DEPS"; exit 9; }
test -f "$PINS/openai/__init__.py" || { echo "FATAL: openai did not land in DEPS"; exit 9; }
echo "vendored pins verified in DEPS"

probe() {
  PYTHONPATH="$PINS:$DEPS" PYTHONWARNINGS=ignore python3 - <<'PY' 2>/dev/null | grep -E "^(IMPORT_OK|MISSING|OTHER)" | head -1
import sys, os
deps = "/iopsstor/scratch/cscs/rosmith/nemo_gym_pins"
sys.path.insert(0, deps)
for m in [m for m in list(sys.modules) if m.split(".")[0] in ("anthropic", "openai")]:
    del sys.modules[m]
try:
    from nemo_gym.cli import RunHelper
    from nemo_gym.rollout_collection import RolloutCollectionHelper
    from nemo_gym.server_utils import BaseServerConfig
    import transformers
    print("IMPORT_OK", transformers.__version__)
except ModuleNotFoundError as e:
    print("MISSING", e.name)
except Exception as e:
    import importlib.util, os as _os
    for _m in [m for m in list(sys.modules) if m.split(".")[0] == "anthropic"]:
        del sys.modules[_m]
    spec = importlib.util.find_spec("anthropic")
    print("OTHER", type(e).__name__, str(e)[:160],
          "| deps_init_exists:", _os.path.exists(deps + "/anthropic/__init__.py"),
          "| find_spec:", spec.origin if spec else None,
          "| path0-2:", sys.path[:3])
PY
}

declare -A PKGMAP=( [yaml]=PyYAML [dotenv]=python-dotenv [openai]="openai<=2.7.2" [anthropic]="anthropic<=0.109.2" [jwt]=PyJWT [git]=GitPython )
for i in $(seq 1 20); do
  OUT=$(probe)
  echo "probe[$i]: $OUT"
  case "$OUT" in
    IMPORT_OK*) touch "$DEPS/.ready"; echo DONE; exit 0 ;;
    MISSING*)
      MOD=$(echo "$OUT" | awk '{print $2}' | cut -d. -f1)
      PKG="${PKGMAP[$MOD]:-$MOD}"
      echo "installing: $PKG"
      pip install -q --target "$DEPS" --no-deps --ignore-installed --upgrade "$PKG"
      ;;
    *) echo "NON-MODULE ERROR — stopping"; exit 2 ;;
  esac
done
echo "loop exhausted"; exit 3
