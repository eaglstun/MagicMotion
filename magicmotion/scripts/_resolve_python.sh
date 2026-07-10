# Sourced by the launch scripts to set $PY to a python that actually has the deps.
#
# Why: on machines where a pyenv (or other) shim shadows conda on PATH, a bare
# `python` can resolve to an interpreter without the pinned stack, and the run dies
# on `ModuleNotFoundError: No module named 'einops'`. Resolve deliberately instead.
#
# Order: $MAGICMOTION_PYTHON override > conda env named "magicmotion" > bare `python`.
PY="${MAGICMOTION_PYTHON:-}"
if [ -z "$PY" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "$CONDA_BASE" ] && [ -x "$CONDA_BASE/envs/magicmotion/bin/python" ]; then
        PY="$CONDA_BASE/envs/magicmotion/bin/python"
    else
        PY="python"
    fi
fi

# Fail fast with a clear message rather than dying deep in an import later.
if ! "$PY" -c "import einops" >/dev/null 2>&1; then
    echo "!! '$PY' is missing 'einops' (wrong interpreter?)." >&2
    echo "   Activate the env (conda activate magicmotion) or set MAGICMOTION_PYTHON to a good python." >&2
    exit 1
fi
