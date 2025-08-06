#!/usr/bin/env bash
set -e
function abort() {
    echo "Error: $1" >&2
    exit 1
}

if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc found: $(command -v nvcc)"
else
    if [ "$EUID" -ne 0 ]; then
    abort "This script must be run as root or with sudo."
    fi
    if ! command -v lspci >/dev/null 2>&1; then
        echo "Info: lspci not found. Installing pciutils to detect NVIDIA GPU..."
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update >/dev/null
            apt-get install -y pciutils
        elif command -v dnf >/dev/null 2>&1; then
            dnf install -y pciutils
        elif command -v yum >/dev/null 2>&1; then
            yum install -y pciutils
        else
            echo "Warning: Could not install pciutils. Cannot verify if an NVIDIA GPU is present."
        fi
    fi
    if ! command -v curl >/dev/null 2>&1; then
        echo "Info: curl not found. It is required for checking Fedora repositories. Installing..."
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update >/dev/null && apt-get install -y curl
        elif command -v dnf >/dev/null 2>&1; then
            dnf install -y curl
        else
            abort "Could not install curl. Please install it manually and re-run the script."
        fi
    fi
    if ! lspci | grep -iq 'NVIDIA'; then
    echo "Warning: No NVIDIA GPU detected in lspci output."
    if [ -z "$CI" ]; then
        read -p "Do you want to continue with the installation anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation aborted by user."
        exit 0
        fi
    else
        echo "CI environment detected. Continuing without prompt."
    fi
    fi
    ARCH=$(uname -m)
    echo "Detected Architecture: $ARCH"
    if [ -f /etc/os-release ]; then
        source /etc/os-release
        DISTRO=$ID
        VERSION_ID=$VERSION_ID
    elif command -v lsb_release >/dev/null 2>&1; then
        DISTRO=$(lsb_release -is | tr '[:upper:]' '[:lower:]')
        VERSION_ID=$(lsb_release -rs)
    else
        abort "Cannot determine Linux distribution."
    fi
    echo "Detected Distribution: $DISTRO $VERSION_ID"
    echo ""
    echo "Attempting to install GCC 14..."
    echo "----------------------------------------------------"
    GCC_INSTALL_COMMANDS=()
    case "$DISTRO" in
        ubuntu|debian)
            GCC_INSTALL_COMMANDS=(
                "apt-get update"
                "apt-get install -y software-properties-common"
                "add-apt-repository ppa:ubuntu-toolchain-r/test -y"
                "apt-get update"
                "apt-get install -y gcc-14 g++-14"
            )
            ;;
        fedora)
            GCC_INSTALL_COMMANDS=(
                "dnf -y install gcc-14 g++-14 || dnf -y install gcc gcc-c++"
            )
            ;;
        rhel|centos|rocky|almalinux)
            echo "Info: For RHEL-based systems, installing GCC 14 requires enabling developer toolsets, which may not be available."
            echo "Attempting to install 'gcc-toolset-14'..."
            RHEL_VERSION=$(echo "$VERSION_ID" | cut -d. -f1)
            if [ "$RHEL_VERSION" = "9" ]; then
                GCC_INSTALL_COMMANDS=(
                    "dnf config-manager --set-enabled crb"
                    "dnf -y install gcc-toolset-14"
                )
            else
                GCC_INSTALL_COMMANDS=(
                    "dnf -y install centos-release-scl"
                    "dnf config-manager --set-enabled powertools || dnf config-manager --set-enabled crb"
                    "dnf -y install gcc-toolset-14"
                )
            fi
            ;;
        *)
            echo "Warning: GCC 14 installation not configured for this OS. Skipping."
            ;;
    esac
    if [ ${#GCC_INSTALL_COMMANDS[@]} -gt 0 ]; then
        for cmd in "${GCC_INSTALL_COMMANDS[@]}"; do
            echo "Executing: $cmd"
            if ! eval "$cmd"; then
                echo "Warning: Command failed: '$cmd'. GCC 14 may not be available for your system."
                echo "Continuing with CUDA installation..."
                break
            fi
        done
        echo "GCC 14 installation process finished."
    else
        echo "Skipping GCC 14 installation for $DISTRO."
    fi
    CUDA_VERSION="12.6"
    CUDA_VERSION_MAJOR_MINOR="12-6"
    INSTALLER_FILENAME=""
    INSTALL_COMMANDS=()
    case "$DISTRO" in
        ubuntu)
            case "$VERSION_ID" in
                22.04)
                    [ "$ARCH" = "x86_64" ] && INSTALLER_FILENAME="cuda-repo-ubuntu2204-12-6-local_12.6.0-1_amd64.deb"
                    [ "$ARCH" = "aarch64" ] && INSTALLER_FILENAME="cuda-repo-ubuntu2204-12-6-local_12.6.0-1_arm64.deb"
                    ;;
                20.04)
                    [ "$ARCH" = "x86_64" ] && INSTALLER_FILENAME="cuda-repo-ubuntu2004-12-6-local_12.6.0-1_amd64.deb"
                    [ "$ARCH" = "aarch64" ] && INSTALLER_FILENAME="cuda-repo-ubuntu2004-12-6-local_12.6.0-1_arm64-sbsa.deb"
                    ;;
                *) abort "Unsupported Ubuntu version: $VERSION_ID" ;;
            esac
            INSTALL_COMMANDS=(
                "apt-get update"
                "apt-get install -y gpgv wget"
                "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//.}/$ARCH/${INSTALLER_FILENAME}"
                "dpkg -i ${INSTALLER_FILENAME}"
                "cp /var/cuda-repo-ubuntu${VERSION_ID//.}-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/"
                "apt-get update"
                "apt-get -y install cuda-toolkit-${CUDA_VERSION_MAJOR_MINOR}"
            )
            ;;
        debian)
            case "$VERSION_ID" in
                12) [ "$ARCH" = "x86_64" ] && INSTALLER_FILENAME="cuda-repo-debian12-12-6-local_12.6.0-1_amd64.deb" ;;
                11) [ "$ARCH" = "x86_64" ] && INSTALLER_FILENAME="cuda-repo-debian11-12-6-local_12.6.0-1_amd64.deb" ;;
                *) abort "Unsupported Debian version: $VERSION_ID" ;;
            esac
            INSTALL_COMMANDS=(
                "apt-get update"
                "apt-get install -y gpgv wget"
                "wget https://developer.download.nvidia.com/compute/cuda/repos/debian${VERSION_ID}/$ARCH/${INSTALLER_FILENAME}"
                "dpkg -i ${INSTALLER_FILENAME}"
                "cp /var/cuda-repo-debian${VERSION_ID}-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/"
                "apt-get update"
                "apt-get -y install cuda-toolkit-${CUDA_VERSION_MAJOR_MINOR}"
            )
            ;;
        fedora)
            FEDORA_VERSION=$VERSION_ID
            [ "$ARCH" != "x86_64" ] && abort "This script currently supports only x86_64 on Fedora."
            if command -v dnf4 >/dev/null 2>&1; then
                DNF_BIN=dnf4
                ADD_REPO_CMD="$DNF_BIN config-manager --add-repo"
            else
                DNF_BIN=dnf
                ADD_REPO_CMD="$DNF_BIN config-manager --add-repo"
            fi
            NVIDIA_URL_BASE="https://developer.download.nvidia.com/compute/cuda/repos"
            CUDA_REPO_URL="${NVIDIA_URL_BASE}/fedora${FEDORA_VERSION}/${ARCH}/cuda-fedora${FEDORA_VERSION}.repo"
            if ! curl --head --silent --fail "$CUDA_REPO_URL" >/dev/null; then
                echo "Info: NVIDIA repo for Fedora ${FEDORA_VERSION} not found. Falling back to a recent, stable version (Fedora 41)."
                CUDA_REPO_URL="${NVIDIA_URL_BASE}/fedora41/${ARCH}/cuda-fedora41.repo"
            fi
            INSTALL_COMMANDS=(
                "$ADD_REPO_CMD $CUDA_REPO_URL"
                "$DNF_BIN clean all"
                "$DNF_BIN -y module disable nvidia-driver"
                "$DNF_BIN -y install cuda-toolkit-${CUDA_VERSION_MAJOR_MINOR}"
            )
            ;;
        rhel|centos|rocky|almalinux)
            RHEL_VERSION=$(echo "$VERSION_ID" | cut -d. -f1)
            [ "$RHEL_VERSION" != "9" ] && [ "$RHEL_VERSION" != "8" ] && abort "Unsupported RHEL/CentOS version: $RHEL_VERSION"
            INSTALL_COMMANDS=(
                "dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${RHEL_VERSION}/$ARCH/cuda-rhel${RHEL_VERSION}.repo"
                "dnf clean all"
                "dnf -y module disable nvidia-driver"
                "dnf -y install cuda-toolkit-${CUDA_VERSION_MAJOR_MINOR}"
            )
            ;;
        *)
            abort "Distribution '$DISTRO' is not supported by this script."
            ;;
    esac
    if [ -z "$INSTALLER_FILENAME" ] && { [ "$DISTRO" = "ubuntu" ] || [ "$DISTRO" = "debian" ]; }; then
        abort "Could not find a suitable installer for $DISTRO $VERSION_ID on $ARCH."
    fi
    echo ""
    echo "Starting CUDA Toolkit ${CUDA_VERSION} installation..."
    echo "----------------------------------------------------"
    for cmd in "${INSTALL_COMMANDS[@]}"; do
        echo "Executing: $cmd"
        if ! eval "$cmd"; then
            if [[ "$DISTRO" == "fedora" && "$cmd" == *config-manager* ]]; then
                echo "Hint: This may mean NVIDIA does not yet provide a repository for Fedora ${VERSION_ID}."
                echo "Please check the NVIDIA CUDA download page for supported versions."
            fi
            abort "Command failed to execute: '$cmd'"
        fi
    done
    echo ""
    echo "Setting up environment variables..."
    ENV_FILE="/etc/profile.d/cuda-${CUDA_VERSION_MAJOR_MINOR}.sh"
    echo "Creating environment file at ${ENV_FILE}"
    tee "$ENV_FILE" > /dev/null <<EOF
    export PATH="/usr/local/cuda-${CUDA_VERSION}/bin:\$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:\$LD_LIBRARY_PATH"
EOF
    if command -v gcc-14 >/dev/null 2>&1 && command -v g++-14 >/dev/null 2>&1; then
        echo "Appending GCC 14 compiler environment variables..."
        {
            echo ""
            echo "export CC=/usr/bin/gcc-14"
            echo "export CXX=/usr/bin/g++-14"
            echo "export CUDAHOSTCXX=/usr/bin/g++-14"
        } >> "$ENV_FILE"
        echo "GCC 14 environment variables exported."
    else
        echo "Warning: gcc-14 and g++-14 not found in PATH; skipping CC/CXX/CUDAHOSTCXX export."
    fi
    if [ -n "$INSTALLER_FILENAME" ] && [ -f "$INSTALLER_FILENAME" ]; then
        echo "Cleaning up installer file: $INSTALLER_FILENAME"
        rm -f "$INSTALLER_FILENAME"
    fi
    echo ""
    echo "----------------------------------------------------"
    echo "CUDA Toolkit ${CUDA_VERSION} installation complete!"
    echo ""
    echo "The environment has been configured in ${ENV_FILE}"
    echo "Verification steps (after shell reload or reboot):"
    echo "  1. nvcc --version"
    echo "  2. nvidia-smi"
    echo "----------------------------------------------------"
fi