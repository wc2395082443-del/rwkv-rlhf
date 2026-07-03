#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env.remote}"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

load_env_file() {
  local path="$1"
  local raw line key value
  [[ -f "$path" ]] || return 0

  while IFS= read -r raw || [[ -n "$raw" ]]; do
    line="$(trim "$raw")"
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ "$line" == export\ * ]]; then
      line="$(trim "${line#export }")"
    fi
    [[ "$line" == *=* ]] || continue

    key="$(trim "${line%%=*}")"
    value="$(trim "${line#*=}")"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -n "${!key+x}" ]] && continue

    if [[ ${#value} -ge 2 && "${value:0:1}" == "${value: -1}" ]] &&
       [[ "${value:0:1}" == "'" || "${value:0:1}" == '"' ]]; then
      value="${value:1:${#value}-2}"
    fi
    export "$key=$value"
  done <"$path"
}

if [[ -f "$ENV_FILE" ]]; then
  load_env_file "$ENV_FILE"
fi

KUBECTL="${KUBECTL:-kubectl}"
DEVPOD="${DEVPOD:-devpod}"
DEVPOD_HOME="${DEVPOD_HOME:-$HOME/.devpod}"
REMOTE_WORKSPACE_ID="${REMOTE_WORKSPACE_ID:-g6}"
REMOTE_NAMESPACE="${REMOTE_NAMESPACE:-devpod}"
REMOTE_POD="${REMOTE_POD:-devpod-default-g6-7378f}"
REMOTE_NODE="${REMOTE_NODE:-g6}"
REMOTE_IMAGE="${REMOTE_IMAGE:-mirrors.bbt.sspu.edu.cn:8081/sspu/bbt-devpod-ubuntu24-cuda:20260525}"
DEVPOD_SEED_IMAGE="${DEVPOD_SEED_IMAGE:-nvcr.io/nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04}"
DEVPOD_SOURCE="${DEVPOD_SOURCE:-$DEVPOD_HOME/manual-sources/$REMOTE_WORKSPACE_ID}"
DEVPOD_POD_TEMPLATE="${DEVPOD_POD_TEMPLATE:-$DEVPOD_HOME/manual-sources/bbt-devpod-shared-mount-template.yaml}"
REMOTE_SSH_HOST="${REMOTE_SSH_HOST:-$REMOTE_WORKSPACE_ID.devpod}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/Projects/MachineLearning/helicopter}"
REMOTE_VENV="${REMOTE_VENV:-$REMOTE_ROOT/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_PROFILE="${INSTALL_PROFILE:-rwkv}"
UPDATE_UV="${UPDATE_UV:-0}"
UV_UPGRADE="${UV_UPGRADE:-0}"
RUN_PIP_CHECK="${RUN_PIP_CHECK:-1}"
UV_SYNC_INEXACT="${UV_SYNC_INEXACT:-1}"
VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-0.11.2.dev278+gdbc3d9991}"
VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-0}"
VLLM_REBUILD="${VLLM_REBUILD:-auto}"
VERL_REINSTALL="${VERL_REINSTALL:-auto}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
BUILD_TMPDIR="${BUILD_TMPDIR:-$REMOTE_ROOT/.tmp}"
REMOTE_HTTP_PROXY="${REMOTE_HTTP_PROXY:-http://192.168.122.1:10810}"
REMOTE_HTTPS_PROXY="${REMOTE_HTTPS_PROXY:-$REMOTE_HTTP_PROXY}"
REMOTE_NO_PROXY="${REMOTE_NO_PROXY:-localhost,127.0.0.1,::1,.svc,.cluster.local,.bbt.sspu.edu.cn,10.0.0.0/8,192.168.0.0/16}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
UV_INDEX_URL="${UV_INDEX_URL:-$PYPI_INDEX_URL}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
CARGO_REGISTRY_MIRROR="${CARGO_REGISTRY_MIRROR:-sparse+https://rsproxy.cn/index/}"
DEVPOD_RECREATE="${DEVPOD_RECREATE:-0}"
SYNC_REMOTE="${SYNC_REMOTE:-1}"
INSTALL_REMOTE="${INSTALL_REMOTE:-1}"

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run() {
  print_cmd "$@"
  [[ "${DRY_RUN:-0}" == "1" ]] || "$@"
}

die() {
  echo "error: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

require_local_tools() {
  have "$KUBECTL" || die "kubectl is required"
  have "$DEVPOD" || die "devpod is required"
  have ssh || die "ssh is required"
  have rsync || die "rsync is required"
}

pod_exists() {
  "$KUBECTL" get pod "$REMOTE_POD" -n "$REMOTE_NAMESPACE" >/dev/null 2>&1
}

pod_spec_image() {
  "$KUBECTL" get pod "$REMOTE_POD" -n "$REMOTE_NAMESPACE" -o jsonpath='{.spec.containers[0].image}'
}

pod_status_image() {
  "$KUBECTL" get pod "$REMOTE_POD" -n "$REMOTE_NAMESPACE" -o jsonpath='{.status.containerStatuses[0].image}'
}

pod_container_id() {
  "$KUBECTL" get pod "$REMOTE_POD" -n "$REMOTE_NAMESPACE" -o jsonpath='{.status.containerStatuses[0].containerID}'
}

devpod_recreate() {
  [[ "$DEVPOD_RECREATE" == "1" ]] || return 0

  run "$DEVPOD" up "$DEVPOD_SOURCE" \
    --id "$REMOTE_WORKSPACE_ID" \
    --recreate \
    --open-ide=false \
    --provider kubernetes \
    --context default \
    --devpod-home "$DEVPOD_HOME" \
    --devcontainer-image "$DEVPOD_SEED_IMAGE" \
    --provider-option "RESOURCES=limits.nvidia.com/gpu=8" \
    --provider-option "KUBERNETES_NAMESPACE=$REMOTE_NAMESPACE" \
    --provider-option "NODE_SELECTOR=kubernetes.io/hostname=$REMOTE_NODE" \
    --provider-option "POD_MANIFEST_TEMPLATE=$DEVPOD_POD_TEMPLATE" \
    --provider-option "KUBERNETES_CONFIG=$HOME/.kube/config" \
    --provider-option "POD_TIMEOUT=10m" \
    --provider-option "STRICT_SECURITY=false"
}

ensure_pod() {
  if ! pod_exists; then
    devpod_recreate
  fi

  pod_exists || die "pod $REMOTE_NAMESPACE/$REMOTE_POD does not exist; set DEVPOD_RECREATE=1 to recreate it"

  local node
  node="$("$KUBECTL" get pod "$REMOTE_POD" -n "$REMOTE_NAMESPACE" -o jsonpath='{.spec.nodeName}')"
  [[ "$node" == "$REMOTE_NODE" ]] || die "pod is on node $node, expected $REMOTE_NODE"
}

wait_for_running_image() {
  local old_container_id="${1:-}"
  local ready status_image new_container_id

  for _ in {1..60}; do
    ready="$("$KUBECTL" get pod "$REMOTE_POD" -n "$REMOTE_NAMESPACE" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || true)"
    status_image="$(pod_status_image 2>/dev/null || true)"
    new_container_id="$(pod_container_id 2>/dev/null || true)"

    if [[ "$ready" == "true" && "$status_image" == "$REMOTE_IMAGE" ]]; then
      if [[ -z "$old_container_id" || "$new_container_id" != "$old_container_id" ]]; then
        return 0
      fi
    fi
    sleep 5
  done

  die "pod did not restart with image $REMOTE_IMAGE"
}

restart_remote_container() {
  local old_container_id
  old_container_id="$(pod_container_id 2>/dev/null || true)"

  print_cmd "$KUBECTL" exec -n "$REMOTE_NAMESPACE" "$REMOTE_POD" -- sh -lc 'kill 1'
  [[ "${DRY_RUN:-0}" == "1" ]] && return 0

  "$KUBECTL" exec -n "$REMOTE_NAMESPACE" "$REMOTE_POD" -- sh -lc 'kill 1' >/dev/null 2>&1 || true
  wait_for_running_image "$old_container_id"
}

ensure_runtime_image() {
  local spec_image status_image
  spec_image="$(pod_spec_image)"
  status_image="$(pod_status_image 2>/dev/null || true)"

  if [[ "$spec_image" != "$REMOTE_IMAGE" ]]; then
    run "$KUBECTL" set image "pod/$REMOTE_POD" -n "$REMOTE_NAMESPACE" "devpod=$REMOTE_IMAGE"
    restart_remote_container
    [[ "${DRY_RUN:-0}" == "1" ]] && return 0
  elif [[ "$status_image" != "$REMOTE_IMAGE" ]]; then
    restart_remote_container
    [[ "${DRY_RUN:-0}" == "1" ]] && return 0
  fi

  run "$KUBECTL" wait -n "$REMOTE_NAMESPACE" --for=condition=Ready "pod/$REMOTE_POD" --timeout=180s

  spec_image="$(pod_spec_image)"
  status_image="$(pod_status_image 2>/dev/null || true)"
  [[ "$spec_image" == "$REMOTE_IMAGE" ]] || die "pod spec image is $spec_image, expected $REMOTE_IMAGE"
  [[ "$status_image" == "$REMOTE_IMAGE" ]] || die "running pod image is $status_image, expected $REMOTE_IMAGE"
}

verify_remote_tools() {
  run "$KUBECTL" exec -n "$REMOTE_NAMESPACE" "$REMOTE_POD" -- bash -lc \
    'set -euo pipefail
     command -v git
     command -v uv
     command -v python3
     command -v cmake
     command -v ninja
     command -v cc
     command -v nvcc
     nvidia-smi -L | wc -l
     test -d /workspace/Projects
     test -d /workspace/Weights
     test -d /workspace/Datasets'
}

sync_remote_repo() {
  [[ "$SYNC_REMOTE" == "1" ]] || return 0

  run ssh "$REMOTE_SSH_HOST" "mkdir -p $(printf '%q' "$REMOTE_ROOT")"
  run rsync -a --delete \
    --exclude '.git/' \
    --exclude '.git' \
    --exclude '.venv/' \
    --exclude '.env' \
    --exclude '.env.local' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '.mypy_cache/' \
    --exclude '.cache/' \
    --exclude '.tmp/' \
    --exclude '.deps/' \
    --exclude '*.so' \
    --exclude 'build/' \
    --exclude 'dist/' \
    --exclude '*.egg-info/' \
    --exclude 'target/' \
    --exclude 'node_modules/' \
    --exclude '/logs/' \
    --exclude '/runs/' \
    --exclude '/outputs/' \
    --exclude '/checkpoints/' \
    --exclude '/wandb/' \
    --exclude '/tensorboard/' \
    --exclude '/weights/' \
    --exclude '/models/' \
    --exclude '/datasets/' \
    --exclude '/data/' \
    "$ROOT/" "$REMOTE_SSH_HOST:$REMOTE_ROOT/"
}

remote_env_args() {
  local args=(
    "PYTHON_VERSION=$PYTHON_VERSION"
    "VENV=$REMOTE_VENV"
    "INSTALL_PROFILE=$INSTALL_PROFILE"
    "INSTALL_SYSTEM_DEPS=0"
    "UPDATE_UV=$UPDATE_UV"
    "UV_UPGRADE=$UV_UPGRADE"
    "RUN_PIP_CHECK=$RUN_PIP_CHECK"
    "UV_SYNC_INEXACT=$UV_SYNC_INEXACT"
    "VLLM_TARGET_DEVICE=$VLLM_TARGET_DEVICE"
    "VLLM_VERSION_OVERRIDE=$VLLM_VERSION_OVERRIDE"
    "VLLM_USE_PRECOMPILED=$VLLM_USE_PRECOMPILED"
    "VLLM_REBUILD=$VLLM_REBUILD"
    "VERL_REINSTALL=$VERL_REINSTALL"
    "CMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE"
    "BUILD_TMPDIR=$BUILD_TMPDIR"
    "HTTP_PROXY=$REMOTE_HTTP_PROXY"
    "HTTPS_PROXY=$REMOTE_HTTPS_PROXY"
    "http_proxy=$REMOTE_HTTP_PROXY"
    "https_proxy=$REMOTE_HTTPS_PROXY"
    "NO_PROXY=$REMOTE_NO_PROXY"
    "no_proxy=$REMOTE_NO_PROXY"
    "ALL_PROXY="
    "all_proxy="
    "PYPI_INDEX_URL=$PYPI_INDEX_URL"
    "UV_INDEX_URL=$UV_INDEX_URL"
    "HF_ENDPOINT=$HF_ENDPOINT"
    "UV_LINK_MODE=$UV_LINK_MODE"
    "CARGO_REGISTRY_MIRROR=$CARGO_REGISTRY_MIRROR"
  )

  printf ' %q' "${args[@]}"
}

install_remote_env() {
  [[ "$INSTALL_REMOTE" == "1" ]] || return 0

  local quoted_root
  quoted_root="$(printf '%q' "$REMOTE_ROOT")"

  run ssh "$REMOTE_SSH_HOST" \
    "cd $quoted_root && env$(remote_env_args) bash scripts/install_local.sh"
}

require_local_tools
ensure_pod
ensure_runtime_image
verify_remote_tools
sync_remote_repo
install_remote_env

echo "Remote environment ready: $REMOTE_SSH_HOST:$REMOTE_ROOT"
