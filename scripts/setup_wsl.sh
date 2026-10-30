#!/usr/bin/env bash
# WSL Ubuntu 24.04 환경 구성 스크립트
#   사용법: bash scripts/setup_wsl.sh [--models] [--kernels] [--train]
set -euo pipefail
cd "$(dirname "$0")/.."

# /mnt/d (NTFS) 위에 venv 를 두면 torch import 가 매우 느리므로 리눅스 FS 에 venv 생성
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robotics-vla}"
export UV_LINK_MODE=copy

# 1) 시스템 패키지 (오프스크린 렌더링용 EGL/OSMesa). 영상 저장은 imageio-ffmpeg 내장 바이너리 사용
if ! dpkg -s libegl1 libosmesa6 libgl1 >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y libegl1 libgl1 libosmesa6 libglfw3 git
fi

# 2) MuJoCo Menagerie - Franka Panda 에셋 (sparse checkout)
if [ ! -f assets/mujoco_menagerie/franka_emika_panda/panda.xml ]; then
  mkdir -p assets
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie
  git -C assets/mujoco_menagerie sparse-checkout set franka_emika_panda
fi

# 3) Python 환경
EXTRAS=()
for a in "$@"; do
  case "$a" in
    --models)  EXTRAS+=(--extra models) ;;
    --kernels) EXTRAS+=(--extra models --extra kernels) ;;
    --train)   EXTRAS+=(--extra models --extra train) ;;
  esac
done
uv sync "${EXTRAS[@]}"

echo
echo "[OK] venv: $UV_PROJECT_ENVIRONMENT"
echo "  아래를 ~/.bashrc 에 추가하면 이후 'uv run' 이 같은 venv 를 사용합니다:"
echo "    export UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
echo "    export UV_LINK_MODE=copy"
echo "    export MUJOCO_GL=egl"
