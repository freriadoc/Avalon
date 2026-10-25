#!/usr/bin/env bash
# Avalon installer (Linux, ROCm): checks the machine, builds llama-server for the detected GPU, downloads the model with
# its MTP draft head, verifies every file against its SHA-256 on Hugging Face and installs an `avalon-server` launcher.
#
#   curl -fsSL https://raw.githubusercontent.com/freriadoc/Avalon/main/install.sh | bash
#   or, from a clone: ./install.sh
#
# Settings (environment): AVALON_DIR (source tree, default ~/Avalon), MODEL_DIR (default
# ~/Avalon-models/Qwen3.8-Flash-Next-Avalon-GGUF), BIN_DIR (launcher, default ~/.local/bin), AMDGPU_TARGETS (skip the
# detection), SKIP_MODEL=1 (build only), FORCE=1 (go on past a failed check), JOBS (build jobs, default nproc).
set -euo pipefail

REPO_URL=https://github.com/freriadoc/Avalon
HF_REPO=freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF
AVALON_DIR=${AVALON_DIR:-$HOME/Avalon}
MODEL_DIR=${MODEL_DIR:-$HOME/Avalon-models/Qwen3.8-Flash-Next-Avalon-GGUF}
BIN_DIR=${BIN_DIR:-$HOME/.local/bin}
JOBS=${JOBS:-$(nproc)}
ROCM=${ROCM_PATH:-/opt/rocm}

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$*" >&2; }
fail() {
    printf '\033[31merror: %s\033[0m\n' "$*" >&2
    if [ "${FORCE:-0}" = 1 ]; then warn "FORCE=1: going on"; else exit 1; fi
}

# ---------------------------------------------------------------------------------------------------------------------
say "Checking the machine"

[ "$(uname -s)" = Linux ] || fail "Linux only (found $(uname -s))"
grep -qw avx2 /proc/cpuinfo || fail "the CPU has no AVX2: the experts computed on the CPU need it"
for tool in git cmake python3; do
    command -v $tool >/dev/null || fail "$tool not found: install it with your package manager"
done

# ROCm
if [ ! -x "$ROCM/bin/amdclang++" ] || [ ! -x "$ROCM/bin/rocminfo" ]; then
    fail "ROCm not found in $ROCM. Install ROCm 7.x first (Ubuntu: https://rocm.docs.amd.com, amdgpu-install), or set ROCM_PATH"
fi
rocm_ver=$(cat "$ROCM/.info/version" 2>/dev/null | cut -d- -f1 || true)
case "$rocm_ver" in
    7.*) echo "ROCm $rocm_ver" ;;
    "")  warn "ROCm version unknown (tested: 7.2.4)" ;;
    *)   fail "ROCm $rocm_ver found; Avalon is tested with ROCm 7.2.4 only" ;;
esac

# GPU: RDNA3 discrete (gfx1100 measured; gfx1101 / gfx1102 should work); APUs and older generations are refused
if [ -z "${AMDGPU_TARGETS:-}" ]; then
    gfx=$("$ROCM/bin/rocminfo" 2>/dev/null | grep -oE '\bgfx[0-9a-f]+\b' | sort -u | tr '\n' ' ' || true)
    for g in gfx1100 gfx1101 gfx1102; do
        case " $gfx " in *" $g "*) AMDGPU_TARGETS=$g; break ;; esac
    done
    if [ -z "${AMDGPU_TARGETS:-}" ]; then
        fail "no RDNA3 discrete GPU found (rocminfo reports: ${gfx:-nothing}). Avalon's kernels target gfx1100 / gfx1101 / gfx1102"
        AMDGPU_TARGETS=${gfx%% *}
    fi
fi
case "$AMDGPU_TARGETS" in
    gfx1100) echo "GPU $AMDGPU_TARGETS (RX 7900 XT / XTX: the measured target)" ;;
    *)       warn "GPU $AMDGPU_TARGETS: should work, not measured (Avalon is tuned on a gfx1100)" ;;
esac

vram_mib=0
for f in /sys/class/drm/card*/device/mem_info_vram_total; do
    [ -r "$f" ] || continue
    v=$(( $(cat "$f") / 1048576 )); [ $v -gt $vram_mib ] && vram_mib=$v
done
if [ $vram_mib -gt 0 ]; then
    echo "VRAM $((vram_mib * 1048576 / 1000000000)) GB"
    [ $vram_mib -ge 19000 ] || warn "less than 20 GB of VRAM: the 256k context with MTP may not fit; try -c 65536"
fi

# RAM and disk
ram_gib=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1048576 ))
echo "RAM $ram_gib GiB"
# The model (~89 GiB) is memory-mapped: what does not fit in RAM is read again from the disk. Measured on the reference
# machine with the server's RAM capped at 28 GB (a 32 GB machine): generation -2 to -6 %, prefill of a new 19k prompt
# -31 % (889 against 1,283 t/s). Not measured below that.
[ $ram_gib -ge 28 ] || fail "$ram_gib GiB of RAM: Avalon is measured with 32 GB or more (the model is memory-mapped, ~89 GiB)"
if [ "${SKIP_MODEL:-0}" != 1 ]; then
    mkdir -p "$MODEL_DIR"
    have=$(du -sk "$MODEL_DIR" 2>/dev/null | cut -f1)
    free=$(( ($(df -Pk "$MODEL_DIR" | awk 'NR == 2 {print $4}') + have) / 1048576 ))
    [ $free -ge 98 ] || fail "$free GiB free for $MODEL_DIR: the model needs ~97 GiB (set MODEL_DIR elsewhere)"
fi

# ---------------------------------------------------------------------------------------------------------------------
say "Getting the source"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)
if [ -n "$script_dir" ] && [ -f "$script_dir/ggml/src/ggml-cuda/moe-cache.cu" ]; then
    AVALON_DIR=$script_dir
    echo "using the clone in $AVALON_DIR"
elif [ -d "$AVALON_DIR/.git" ]; then
    echo "updating $AVALON_DIR"
    git -C "$AVALON_DIR" pull --ff-only
else
    git clone --depth 1 "$REPO_URL" "$AVALON_DIR"
fi

# ---------------------------------------------------------------------------------------------------------------------
say "Building llama-server for $AMDGPU_TARGETS (10 – 30 minutes)"

cmake -S "$AVALON_DIR" -B "$AVALON_DIR/build" -DGGML_HIP=ON -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$ROCM/bin/amdclang" -DCMAKE_CXX_COMPILER="$ROCM/bin/amdclang++" \
    > "$AVALON_DIR/build-configure.log" 2>&1 || { tail -20 "$AVALON_DIR/build-configure.log"; fail "cmake failed (log: $AVALON_DIR/build-configure.log)"; }
cmake --build "$AVALON_DIR/build" -j"$JOBS" --target llama-server > "$AVALON_DIR/build.log" 2>&1 \
    || { tail -30 "$AVALON_DIR/build.log"; fail "build failed (log: $AVALON_DIR/build.log)"; }
SERVER=$AVALON_DIR/build/bin/llama-server
echo "built $SERVER"

# ---------------------------------------------------------------------------------------------------------------------
if [ "${SKIP_MODEL:-0}" != 1 ]; then
    say "Downloading the model (~89 GiB, resumable: run the installer again if it stops)"

    VENV=$AVALON_DIR/.venv
    [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV" || fail "python3 -m venv failed (Debian / Ubuntu: apt install python3-venv)"
    "$VENV/bin/pip" install -q --upgrade "huggingface_hub>=0.34"
    "$VENV/bin/hf" download "$HF_REPO" --local-dir "$MODEL_DIR" --exclude '*.jpg' --exclude '*.md'

    say "Verifying the SHA-256 of every file"
    "$VENV/bin/python" - "$HF_REPO" "$MODEL_DIR" <<'EOF'
import hashlib, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
repo, root = sys.argv[1], sys.argv[2]
tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"))
files = [(f["path"], f["lfs"]["oid"], f["size"]) for f in tree if f["type"] == "file" and f.get("lfs") and not f["path"].endswith(".jpg")]
done_path = os.path.join(root, ".avalon-verified")
done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
def check(item):
    path, oid, size = item
    p = os.path.join(root, path)
    if not os.path.exists(p) or os.path.getsize(p) != size:
        return path, "missing or incomplete"
    key = f"{path}:{oid}:{int(os.path.getmtime(p))}"
    if key in done:
        return path, key
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return path, key if h.hexdigest() == oid else "SHA-256 mismatch"
bad = []
with ThreadPoolExecutor(max(1, min(8, os.cpu_count() or 1))) as ex:
    for path, r in ex.map(check, files):
        if ":" in r:
            done.add(r)
        else:
            bad.append(path); print(f"  {path}: {r}")
open(done_path, "w").write("\n".join(sorted(done)) + "\n")
if bad:
    for path in bad:
        try: os.remove(os.path.join(root, path))
        except FileNotFoundError: pass
    sys.exit(f"{len(bad)} file(s) removed: run the installer again to download them")
print(f"  {len(files)} files OK")
EOF
fi

# ---------------------------------------------------------------------------------------------------------------------
say "Installing the launcher $BIN_DIR/avalon-server"

mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/avalon-server" <<EOF
#!/usr/bin/env bash
# Avalon: Qwen3.8-Flash-Next with MTP, 256k context, OpenAI-compatible API on http://\${HOST:-127.0.0.1}:\${PORT:-8080}
# Extra arguments go to llama-server (e.g. -c 65536 for a smaller context).
cd "$MODEL_DIR"
exec "$SERVER" -m Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf \\
  -c 262144 -b 8192 -ub 8192 -fa on -ctk q8_0 -ctv q4_0 --jinja --parallel 1 \\
  -md MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf --spec-type draft-mtp \\
  --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0 \\
  --host "\${HOST:-127.0.0.1}" --port "\${PORT:-8080}" "\$@"
EOF
chmod +x "$BIN_DIR/avalon-server"
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) warn "$BIN_DIR is not in your PATH" ;; esac

say "Done. Start it with: avalon-server   (then open http://127.0.0.1:8080)"
