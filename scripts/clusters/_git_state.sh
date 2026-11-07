#!/bin/bash

ZPUFFER_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && git rev-parse --show-toplevel) || {
    echo "Unable to locate the git repository" >&2
    return 1 2>/dev/null || exit 1
}

zpuffer_snapshot() {
    local git_dir
    local snapshot_dir
    local capture_diff=0

    git_dir=$(git -C "$ZPUFFER_REPO_ROOT" rev-parse --git-common-dir) || return 1
    if [[ "$git_dir" != /* ]]; then
        git_dir="$ZPUFFER_REPO_ROOT/$git_dir"
    fi
    snapshot_dir="$git_dir/zpuffer_snapshots"
    mkdir -p "$snapshot_dir" || return 1

    if [[ -z "${ZPUFFER_COMMIT:-}" ]]; then
        ZPUFFER_COMMIT=$(git -C "$ZPUFFER_REPO_ROOT" rev-parse HEAD) || return 1
        capture_diff=1
    else
        ZPUFFER_COMMIT=$(git -C "$ZPUFFER_REPO_ROOT" rev-parse --verify "${ZPUFFER_COMMIT}^{commit}") || {
            echo "Unknown commit: $ZPUFFER_COMMIT" >&2
            return 1
        }
    fi

    if [[ -z "${ZPUFFER_DIFF_FILE:-}" ]]; then
        ZPUFFER_DIFF_FILE="$snapshot_dir/git-diff-$ZPUFFER_COMMIT-$(date +%Y%m%d_%H%M%S)-$$.patch"
        if [[ "$capture_diff" == 1 ]]; then
            git -C "$ZPUFFER_REPO_ROOT" diff --binary HEAD > "$ZPUFFER_DIFF_FILE" || return 1
        else
            : > "$ZPUFFER_DIFF_FILE" || return 1
        fi
    fi
    [[ -f "$ZPUFFER_DIFF_FILE" ]] || {
        echo "Diff file not found: $ZPUFFER_DIFF_FILE" >&2
        return 1
    }
    export ZPUFFER_COMMIT ZPUFFER_DIFF_FILE
}

zpuffer_submit_with_snapshot() {
    local script=$1
    shift

    if [[ -z "${SLURM_JOB_ID:-}" ]] && command -v sbatch >/dev/null; then
        zpuffer_snapshot || exit 1
        echo "Submitting with commit $ZPUFFER_COMMIT and diff $ZPUFFER_DIFF_FILE"
        exec sbatch --export=ALL "$script" "$@"
    fi
}

zpuffer_cleanup_workdir() {
    if [[ -n "${ZPUFFER_WORKDIR:-}" && "$ZPUFFER_WORKDIR" != "$ZPUFFER_REPO_ROOT" ]]; then
        git -C "$ZPUFFER_REPO_ROOT" worktree remove --force "$ZPUFFER_WORKDIR" || return 1
        rmdir "$(dirname "$ZPUFFER_WORKDIR")" || return 1
    fi
}

zpuffer_record_git_state() {
    local output_dir=$1
    ZPUFFER_GIT_STATE_DIR="$output_dir/git-state/${SLURM_JOB_ID:-local-$$}"
    mkdir -p "$ZPUFFER_GIT_STATE_DIR" || return 1
    cp "$ZPUFFER_DIFF_FILE" "$ZPUFFER_GIT_STATE_DIR/diff.patch" || return 1
    printf '%s\n' "$ZPUFFER_COMMIT" > "$ZPUFFER_GIT_STATE_DIR/commit.txt" || return 1
    printf '%s\n' "${SLURM_JOB_ID:-}" > "$ZPUFFER_GIT_STATE_DIR/slurm-job-id.txt" || return 1
    echo "Git state: commit $ZPUFFER_COMMIT, files in $ZPUFFER_GIT_STATE_DIR"
    export ZPUFFER_GIT_STATE_DIR
}

zpuffer_prepare_workdir() {
    local commit=$1
    local diff_file=$2
    local worktree_parent

    if [[ -z "$commit" ]]; then
        ZPUFFER_COMMIT=""
        ZPUFFER_WORKDIR=$ZPUFFER_REPO_ROOT
        export ZPUFFER_COMMIT ZPUFFER_WORKDIR
        return 0
    fi

    ZPUFFER_COMMIT=$commit
    ZPUFFER_DIFF_FILE=$diff_file
    zpuffer_snapshot || return 1
    worktree_parent=$(mktemp -d "${TMPDIR:-/tmp}/zpufferdrive-worktree.XXXXXX") || return 1
    ZPUFFER_WORKDIR="$worktree_parent/worktree"
    if ! git -C "$ZPUFFER_REPO_ROOT" worktree add --detach "$ZPUFFER_WORKDIR" "$ZPUFFER_COMMIT"; then
        rmdir "$worktree_parent"
        return 1
    fi
    trap zpuffer_cleanup_workdir EXIT
    if [[ -s "$ZPUFFER_DIFF_FILE" ]]; then
        git -C "$ZPUFFER_WORKDIR" apply --binary "$ZPUFFER_DIFF_FILE" || {
            echo "Unable to apply $ZPUFFER_DIFF_FILE on $ZPUFFER_COMMIT" >&2
            return 1
        }
    fi
    export ZPUFFER_COMMIT ZPUFFER_WORKDIR ZPUFFER_DIFF_FILE
}

