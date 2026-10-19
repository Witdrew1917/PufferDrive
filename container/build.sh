#!/bin/bash
# Build the PufferDrive Singularity/Apptainer image.
#
#   ./container/build.sh                  # -> container/pufferdrive.sif
#   ./container/build.sh --sandbox        # writable dir, for iterating on the def
#   ./container/build.sh -o "$PROJ/pufferdrive.sif" --tmpdir "$PROJ/tmp"
#
# Unprivileged only -- never calls sudo. Uses --fakeroot (or --remote).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEF_FILE="$SCRIPT_DIR/pufferdrive.def"
OUTPUT=""
SANDBOX=0
FORCE=0
REMOTE=0
CACHEDIR_OPT=""
TMPDIR_OPT=""

usage() {
    sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    cat <<'EOF'

Options:
  -d, --def FILE     definition file (default: container/pufferdrive.def)
  -o, --output PATH  image path (default: container/pufferdrive.sif)
  -s, --sandbox      build a writable sandbox directory instead of a .sif
  -f, --force        overwrite an existing image
  -r, --remote       use the remote builder (no local root needed)
      --cachedir DIR image layer cache (~5G; default $APPTAINER_CACHEDIR
                     or ~/.cache/apptainer)
      --tmpdir DIR   build scratch space (~15G; default $APPTAINER_TMPDIR,
                     $TMPDIR, then /var/tmp/$USER)
  -h, --help         show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--def)     DEF_FILE="$2"; shift 2 ;;
        -o|--output)  OUTPUT="$2";   shift 2 ;;
        -s|--sandbox) SANDBOX=1;     shift ;;
        -f|--force)   FORCE=1;       shift ;;
        -r|--remote)  REMOTE=1;      shift ;;
        --cachedir)   CACHEDIR_OPT="$2"; shift 2 ;;
        --tmpdir)     TMPDIR_OPT="$2";   shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "error: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
    esac
done

# --- locate the container runtime -------------------------------------------
if command -v apptainer >/dev/null 2>&1; then
    RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
    RUNTIME=singularity
else
    echo "error: neither apptainer nor singularity found on PATH" >&2
    exit 1
fi

[[ -f "$DEF_FILE" ]] || { echo "error: no such def file: $DEF_FILE" >&2; exit 1; }

if [[ -z "$OUTPUT" ]]; then
    if (( SANDBOX )); then
        OUTPUT="$SCRIPT_DIR/pufferdrive.sandbox"
    else
        OUTPUT="$SCRIPT_DIR/pufferdrive.sif"
    fi
fi

if [[ -e "$OUTPUT" && $FORCE -eq 0 ]]; then
    echo "error: $OUTPUT already exists (use --force to overwrite)" >&2
    exit 1
fi

# --- assemble the build command ---------------------------------------------
CMD=("$RUNTIME" build)
(( FORCE ))   && CMD+=(--force)
(( SANDBOX )) && CMD+=(--sandbox)

# This script never escalates privileges: unprivileged builds only.
if (( REMOTE )); then
    CMD+=(--remote)
    echo ">>> building on the remote builder"
elif [[ "$(id -u)" -eq 0 ]]; then
    echo ">>> already running as root"
elif "$RUNTIME" build --help 2>/dev/null | grep -q -- --fakeroot; then
    CMD+=(--fakeroot)
    echo ">>> building with --fakeroot (unprivileged)"

    # --fakeroot needs either subuid/subgid ranges or the fakeroot(1) helper.
    if ! grep -q "^$(id -un):" /etc/subuid 2>/dev/null \
       && ! command -v fakeroot >/dev/null 2>&1; then
        cat >&2 <<'EOF'
error: unprivileged --fakeroot is not usable here.

The def file runs apt-get, which needs root inside the build namespace. Pick one:
  * install the fakeroot helper:  apt-get install fakeroot
  * ask an admin for subuid/subgid ranges in /etc/subuid and /etc/subgid
  * build elsewhere and copy the .sif over, or use --remote
EOF
        exit 1
    fi
else
    cat >&2 <<EOF
error: $RUNTIME has no --fakeroot support and you are not root.

This script will not call sudo. Build the image on a machine where you have
$RUNTIME with --fakeroot (or root), then copy the .sif to this host, or
re-run with --remote to use the remote builder.
EOF
    exit 1
fi

CMD+=("$OUTPUT" "$DEF_FILE")

# Keep the (large) CUDA layer cache and build bundle off /tmp, which is often
# small on HPC nodes. Both need real space: ~5G for the cache, ~15G for the
# bundle. Candidates are tried in order; the first writable one wins, so an
# inherited-but-unusable TMPDIR (e.g. a /scratch path that only exists on
# compute nodes) degrades to a warning instead of killing the build.
pick_dir() {
    local label="$1"; shift
    local d
    for d in "$@"; do
        [[ -n "$d" ]] || continue
        if mkdir -p "$d" 2>/dev/null && [[ -w "$d" ]]; then
            printf '%s' "$d"
            return 0
        fi
        echo ">>> warning: $label '$d' is not writable, trying next" >&2
    done
    echo "error: no writable $label found; pass --cachedir/--tmpdir" >&2
    return 1
}

CACHEDIR="$(pick_dir cachedir \
    "$CACHEDIR_OPT" "${APPTAINER_CACHEDIR:-}" "${SINGULARITY_CACHEDIR:-}" \
    "$HOME/.cache/apptainer" "/var/tmp/$(id -un)/apptainer-cache")"
TMPDIR_BUILD="$(pick_dir tmpdir \
    "$TMPDIR_OPT" "${APPTAINER_TMPDIR:-}" "${SINGULARITY_TMPDIR:-}" "${TMPDIR:-}" \
    "/var/tmp/$(id -un)" "$HOME/.apptainer-tmp")"

export APPTAINER_CACHEDIR="$CACHEDIR"
export SINGULARITY_CACHEDIR="$CACHEDIR"
export APPTAINER_TMPDIR="$TMPDIR_BUILD"
export SINGULARITY_TMPDIR="$TMPDIR_BUILD"

# The build bundle is unpacked uncompressed; warn early rather than dying at 90%.
avail_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
for d in "$CACHEDIR" "$TMPDIR_BUILD"; do
    a="$(avail_gb "$d")"
    if [[ -n "$a" && "$a" -lt 20 ]]; then
        echo ">>> warning: only ${a}G free on $d (want ~20G for a CUDA devel build)" >&2
    fi
done

echo ">>> runtime : $RUNTIME"
echo ">>> def     : $DEF_FILE"
echo ">>> output  : $OUTPUT"
echo ">>> cachedir: $APPTAINER_CACHEDIR"
echo ">>> tmpdir  : $APPTAINER_TMPDIR"
echo ">>> ${CMD[*]}"
echo

"${CMD[@]}"

cat <<EOF

Built $OUTPUT

Next, create the venv and compile the C extensions against your checkout:

    cd $REPO_ROOT
    $RUNTIME exec --nv --bind \$PWD:\$PWD --pwd \$PWD \\
        $OUTPUT /opt/pufferdrive/setup-venv.sh

Then run things with the venv auto-activated:

    $RUNTIME run --nv --bind \$PWD:\$PWD --pwd \$PWD \\
        $OUTPUT puffer train puffer_drive
EOF
