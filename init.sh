#!/usr/bin/env bash
set -e

abort() {
  echo "ERROR: $1" >&2
  exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [ "$(id -u)" -ne 0 ]; then
    echo "This script requires root privileges. Re-executing with sudo..."
    exec sudo -- "./$0" "$@"
fi

if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
  ORIGINAL_USER="$SUDO_USER"
else
  ORIGINAL_USER=$USER 
fi

USER_HOME=$(getent passwd "$ORIGINAL_USER" | cut -d: -f6)
ORIGINAL_SHELL=$(getent passwd "$ORIGINAL_USER" | cut -d: -f7)
echo "--- Running setup for user: $ORIGINAL_USER ---"

if command -v curl >/dev/null 2>&1; then
  DOWNLOAD_CMD=("curl" "-fsSL" "-o")
elif command -v wget >/dev/null 2>&1; then
  DOWNLOAD_CMD=("wget" "-q" "-O")
else
  echo "curl/wget not found. Attempting to install curl..."
  if command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y curl
  elif command -v dnf >/dev/null 2>&1; then dnf install -y curl
  elif command -v yum >/dev/null 2>&1; then yum install -y curl
  else abort "Could not find a supported package manager to install curl."; fi
  DOWNLOAD_CMD=("curl" "-fsSL" "-o")
fi

CONDA_INSTALL_PATH="$USER_HOME/miniconda3"
CONDA_EXEC_PATH="$CONDA_INSTALL_PATH/bin/conda"

if [ ! -f "$CONDA_EXEC_PATH" ]; then
    echo "Conda not found. Installing Miniconda for '$ORIGINAL_USER'..."
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64) MC_FILE="Miniconda3-latest-Linux-x86_64.sh";;
        aarch64) MC_FILE="Miniconda3-latest-Linux-aarch64.sh";;
        *) abort "Unsupported architecture: $ARCH";;
    esac
    MC_URL="https://repo.anaconda.com/miniconda/$MC_FILE"
    
    TMP_DIR=$(sudo -u "$ORIGINAL_USER" mktemp -d)

    echo "Downloading Miniconda installer..."
    sudo -u "$ORIGINAL_USER" "${DOWNLOAD_CMD[@]}" "$TMP_DIR/miniconda.sh" "$MC_URL" || abort "Miniconda download failed."

    echo "Installing Miniconda to $CONDA_INSTALL_PATH..."
    sudo -u "$ORIGINAL_USER" bash "$TMP_DIR/miniconda.sh" -b -p "$CONDA_INSTALL_PATH" || abort "Miniconda installation failed."
    rm -rf "$TMP_DIR"
    echo "Miniconda installation complete."
else
    echo "Miniconda already found at $CONDA_INSTALL_PATH."
fi

if [ ! -L /usr/local/bin/conda ]; then
    echo "Creating symlink for conda at /usr/local/bin/conda..."
    ln -s "$CONDA_EXEC_PATH" /usr/local/bin/conda
else
    echo "Symlink /usr/local/bin/conda already exists."
fi

echo "Initializing Conda for user's shell environment..."
sudo -u "$ORIGINAL_USER" -i -- "$CONDA_EXEC_PATH" init bash zsh

CONDA_BASE_PATH=$(sudo -u "$ORIGINAL_USER" -i -- "$CONDA_EXEC_PATH" info --base)
CONDA_SH_PATH="$CONDA_BASE_PATH/etc/profile.d/conda.sh"

if [ ! -f "$CONDA_SH_PATH" ]; then
    abort "Could not find conda.sh. Conda initialization may have failed."
fi

run_in_conda_shell() {
    sudo -u "$ORIGINAL_USER" -i bash -c ". '$CONDA_SH_PATH' && $1"
}

ENV_NAME="yolov9"

echo "Accepting Conda Terms of Service..."
run_in_conda_shell "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main"
run_in_conda_shell "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r"
run_in_conda_shell "conda tos accept --channel defaults"

if ! run_in_conda_shell "conda env list | grep -q '^$ENV_NAME\s'"; then
  echo "Creating conda environment '$ENV_NAME' with Python 3.10..."
  run_in_conda_shell "conda create -y -n $ENV_NAME python=3.10" || abort "Failed to create conda environment '$ENV_NAME'."
else
  echo "Conda environment '$ENV_NAME' already exists."
fi

RC_FILE="$USER_HOME/.bashrc"
if [[ "$ORIGINAL_SHELL" == *zsh* ]]; then
  RC_FILE="$USER_HOME/.zshrc"
fi

ACTIVATION_LINE="conda activate $ENV_NAME"
if ! grep -Fxq "$ACTIVATION_LINE" "$RC_FILE" 2>/dev/null; then
  echo "Adding '$ACTIVATION_LINE' to $RC_FILE to make it the default for new shells."
  echo -e "\n# Automatically activate the $ENV_NAME environment\n$ACTIVATION_LINE" | tee -a "$RC_FILE" > /dev/null
else
  echo "'$ACTIVATION_LINE' already present in $RC_FILE."
fi

REQUIREMENTS_FILE="$script_dir/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    sudo -u "$ORIGINAL_USER" -i -- "$CONDA_EXEC_PATH" run --no-capture-output -n "$ENV_NAME" pip install uv
    echo "Installing Python dependencies from requirements.txt..."
    if command -v /home/ubuntu/.local/bin/uv >/dev/null 2>&1; then
        sudo -u "$ORIGINAL_USER" -i -- "$CONDA_EXEC_PATH" run --no-capture-output -n "$ENV_NAME" \
            /home/ubuntu/.local/bin/uv pip install -r /home/ubuntu/yolov9001/pyproject.toml --only-binary=:all: \
        || abort "uv pip install failed."
    else
        sudo -u "$ORIGINAL_USER" -i -- "$CONDA_EXEC_PATH" run --no-capture-output -n "$ENV_NAME" \
            python -m pip install -r "$REQUIREMENTS_FILE" --only-binary=:all: \
        || abort "pip install failed."
    fi
    echo "Python dependencies installed."
else
    echo "Warning: requirements.txt not found at '$script_dir/requirements.txt'. Skipping dependency installation."
fi

echo "--- Setup complete! ---"
