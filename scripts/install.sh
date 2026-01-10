#!/usr/bin/env bash
#
# PolyglotLink Installer for Linux/macOS
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/polyglotlink/polyglotlink/main/scripts/install.sh | bash
#
# Or download and run:
#   chmod +x install.sh
#   ./install.sh
#
# Options:
#   --dev       Install development dependencies
#   --docker    Set up Docker environment
#   --systemd   Install as systemd service
#   --uninstall Remove PolyglotLink
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
INSTALL_DIR="${POLYGLOTLINK_INSTALL_DIR:-$HOME/.polyglotlink}"
BIN_DIR="${POLYGLOTLINK_BIN_DIR:-$HOME/.local/bin}"
CONFIG_DIR="${POLYGLOTLINK_CONFIG_DIR:-$HOME/.config/polyglotlink}"
VERSION="${POLYGLOTLINK_VERSION:-latest}"
REPO_URL="https://github.com/polyglotlink/polyglotlink"
MIN_PYTHON_VERSION="3.10"

# Parse arguments
DEV_INSTALL=false
DOCKER_SETUP=false
SYSTEMD_INSTALL=false
UNINSTALL=false

for arg in "$@"; do
    case $arg in
        --dev)
            DEV_INSTALL=true
            ;;
        --docker)
            DOCKER_SETUP=true
            ;;
        --systemd)
            SYSTEMD_INSTALL=true
            ;;
        --uninstall)
            UNINSTALL=true
            ;;
        --help|-h)
            echo "PolyglotLink Installer"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dev       Install development dependencies"
            echo "  --docker    Set up Docker environment"
            echo "  --systemd   Install as systemd service"
            echo "  --uninstall Remove PolyglotLink"
            echo "  --help      Show this help message"
            exit 0
            ;;
    esac
done

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo ""
    echo -e "${BLUE}"
    echo "  ____       _             _       _   _     _       _    "
    echo " |  _ \ ___ | |_   _  __ _| | ___ | |_| |   (_)_ __ | | __"
    echo " | |_) / _ \| | | | |/ _\` | |/ _ \| __| |   | | '_ \| |/ /"
    echo " |  __/ (_) | | |_| | (_| | | (_) | |_| |___| | | | |   < "
    echo " |_|   \___/|_|\__, |\__, |_|\___/ \__|_____|_|_| |_|_|\_\\"
    echo "               |___/ |___/                                "
    echo -e "${NC}"
    echo " Semantic API Translator for IoT Device Ecosystems"
    echo ""
}

check_os() {
    case "$(uname -s)" in
        Linux*)
            OS="linux"
            ;;
        Darwin*)
            OS="macos"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            log_error "Windows detected. Please use install.bat instead."
            exit 1
            ;;
        *)
            log_error "Unsupported operating system: $(uname -s)"
            exit 1
            ;;
    esac
    log_info "Detected OS: $OS"
}

check_python() {
    log_info "Checking Python version..."

    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed."
        log_info "Install Python 3.10+ from https://www.python.org/downloads/"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

    if [[ "$(printf '%s\n' "$MIN_PYTHON_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$MIN_PYTHON_VERSION" ]]; then
        log_error "Python $MIN_PYTHON_VERSION+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi

    log_success "Python $PYTHON_VERSION found"
}

check_dependencies() {
    log_info "Checking dependencies..."

    local missing=()

    if ! command -v pip3 &> /dev/null; then
        missing+=("pip3")
    fi

    if ! command -v git &> /dev/null; then
        missing+=("git")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_info "Please install them and try again."
        exit 1
    fi

    log_success "All dependencies found"
}

create_directories() {
    log_info "Creating directories..."

    mkdir -p "$INSTALL_DIR"
    mkdir -p "$BIN_DIR"
    mkdir -p "$CONFIG_DIR"

    log_success "Directories created"
}

install_polyglotlink() {
    log_info "Installing PolyglotLink..."

    # Create virtual environment
    python3 -m venv "$INSTALL_DIR/venv"
    source "$INSTALL_DIR/venv/bin/activate"

    # Upgrade pip
    pip install --upgrade pip

    # Install PolyglotLink
    if [[ "$DEV_INSTALL" == true ]]; then
        pip install "polyglotlink[dev,test]"
    else
        pip install polyglotlink
    fi

    deactivate

    log_success "PolyglotLink installed"
}

create_wrapper_script() {
    log_info "Creating wrapper script..."

    cat > "$BIN_DIR/polyglotlink" << EOF
#!/usr/bin/env bash
# PolyglotLink wrapper script
source "$INSTALL_DIR/venv/bin/activate"
python -m polyglotlink.app.main "\$@"
EOF

    chmod +x "$BIN_DIR/polyglotlink"

    log_success "Wrapper script created at $BIN_DIR/polyglotlink"
}

create_default_config() {
    log_info "Creating default configuration..."

    if [[ ! -f "$CONFIG_DIR/config.yaml" ]]; then
        cat > "$CONFIG_DIR/config.yaml" << 'EOF'
# PolyglotLink Configuration
# See documentation for all options

environment: development

logging:
  level: INFO
  format: console

http:
  enabled: true
  host: "127.0.0.1"
  port: 8080

mqtt:
  enabled: false
  broker_host: localhost
  broker_port: 1883

redis:
  host: localhost
  port: 6379
EOF
        log_success "Default configuration created at $CONFIG_DIR/config.yaml"
    else
        log_info "Configuration already exists, skipping"
    fi
}

create_env_file() {
    log_info "Creating environment file..."

    if [[ ! -f "$CONFIG_DIR/.env" ]]; then
        cat > "$CONFIG_DIR/.env" << 'EOF'
# PolyglotLink Environment Variables
# Copy this file and fill in your values

# Required for semantic translation
OPENAI_API_KEY=

# Environment
POLYGLOTLINK_ENV=development

# Logging
LOG_LEVEL=INFO
EOF
        log_success "Environment file created at $CONFIG_DIR/.env"
        log_warning "Please edit $CONFIG_DIR/.env and add your API keys"
    else
        log_info "Environment file already exists, skipping"
    fi
}

setup_docker() {
    if [[ "$DOCKER_SETUP" != true ]]; then
        return
    fi

    log_info "Setting up Docker environment..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        return
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_warning "docker-compose not found, trying 'docker compose'..."
    fi

    # Pull the image
    docker pull ghcr.io/polyglotlink/polyglotlink:latest

    log_success "Docker setup complete"
}

install_systemd_service() {
    if [[ "$SYSTEMD_INSTALL" != true ]]; then
        return
    fi

    if [[ "$OS" != "linux" ]]; then
        log_warning "Systemd is only available on Linux"
        return
    fi

    log_info "Installing systemd service..."

    sudo tee /etc/systemd/system/polyglotlink.service > /dev/null << EOF
[Unit]
Description=PolyglotLink - Semantic API Translator for IoT
After=network.target redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin"
EnvironmentFile=$CONFIG_DIR/.env
ExecStart=$INSTALL_DIR/venv/bin/python -m polyglotlink.app.main serve
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable polyglotlink

    log_success "Systemd service installed"
    log_info "Start with: sudo systemctl start polyglotlink"
}

update_path() {
    log_info "Updating PATH..."

    local shell_rc=""
    case "$SHELL" in
        */bash)
            shell_rc="$HOME/.bashrc"
            ;;
        */zsh)
            shell_rc="$HOME/.zshrc"
            ;;
        */fish)
            shell_rc="$HOME/.config/fish/config.fish"
            ;;
        *)
            log_warning "Unknown shell, please add $BIN_DIR to your PATH manually"
            return
            ;;
    esac

    if [[ -f "$shell_rc" ]]; then
        if ! grep -q "POLYGLOTLINK" "$shell_rc"; then
            echo "" >> "$shell_rc"
            echo "# PolyglotLink" >> "$shell_rc"
            echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$shell_rc"
            log_success "Added $BIN_DIR to PATH in $shell_rc"
        fi
    fi
}

uninstall() {
    log_info "Uninstalling PolyglotLink..."

    # Stop service if running
    if [[ -f /etc/systemd/system/polyglotlink.service ]]; then
        sudo systemctl stop polyglotlink || true
        sudo systemctl disable polyglotlink || true
        sudo rm /etc/systemd/system/polyglotlink.service
        sudo systemctl daemon-reload
    fi

    # Remove files
    rm -rf "$INSTALL_DIR"
    rm -f "$BIN_DIR/polyglotlink"

    log_success "PolyglotLink uninstalled"
    log_info "Configuration preserved at $CONFIG_DIR"
    log_info "Remove manually with: rm -rf $CONFIG_DIR"
}

verify_installation() {
    log_info "Verifying installation..."

    if "$BIN_DIR/polyglotlink" version &> /dev/null; then
        VERSION=$("$BIN_DIR/polyglotlink" version 2>/dev/null || echo "unknown")
        log_success "Installation verified: $VERSION"
    else
        log_warning "Could not verify installation"
        log_info "You may need to restart your shell or run: source ~/.bashrc"
    fi
}

print_next_steps() {
    echo ""
    echo -e "${GREEN}Installation complete!${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Restart your shell or run:"
    echo "     source ~/.bashrc  # or ~/.zshrc"
    echo ""
    echo "  2. Configure your API key:"
    echo "     export OPENAI_API_KEY=your-key-here"
    echo ""
    echo "  3. Start the server:"
    echo "     polyglotlink serve"
    echo ""
    echo "  4. Check health:"
    echo "     curl http://localhost:8080/health"
    echo ""
    echo "Documentation: https://polyglotlink.io/docs"
    echo "Issues: https://github.com/polyglotlink/polyglotlink/issues"
    echo ""
}

# Main
print_banner

if [[ "$UNINSTALL" == true ]]; then
    uninstall
    exit 0
fi

check_os
check_python
check_dependencies
create_directories
install_polyglotlink
create_wrapper_script
create_default_config
create_env_file
setup_docker
install_systemd_service
update_path
verify_installation
print_next_steps
