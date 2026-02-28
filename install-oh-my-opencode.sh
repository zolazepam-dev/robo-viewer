#!/bin/bash

# Oh My OpenCode Full Installation Script
# This script automates the complete installation process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root. Please run as a regular user."
        exit 1
    fi
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check if curl is available
    if ! command -v curl &> /dev/null; then
        log_error "curl is required but not installed. Please install curl and try again."
        exit 1
    fi
    
    # Check if npm is available
    if ! command -v npm &> /dev/null; then
        log_warning "npm not found. Will try to install using npm if available."
    fi
    
    # Check if bun is available
    if ! command -v bun &> /dev/null; then
        log_warning "bun not found. Will try to install using npm if available."
    fi
    
    # Check if node is available
    if ! command -v node &> /dev/null; then
        log_warning "Node.js not found. Some installation methods may not work."
    fi
}

# Install OpenCode
install_opencode() {
    log_info "Installing OpenCode..."
    
    if command -v opencode &> /dev/null; then
        log_success "OpenCode is already installed"
        return 0
    fi
    
    # Try the official installer first
    if curl -fsSL https://opencode.ai/install.sh | sh; then
        log_success "OpenCode installed successfully"
        return 0
    else
        log_error "Failed to install OpenCode using official installer"
        return 1
    fi
}

# Install Oh My OpenCode using the recommended method
install_ohmyopencode() {
    log_info "Installing Oh My OpenCode..."
    
    # Try bunx first (recommended)
    if command -v bunx &> /dev/null; then
        log_info "Using bunx (recommended method)..."
        if bunx oh-my-opencode install; then
            log_success "Oh My OpenCode installed successfully using bunx"
            return 0
        fi
    fi
    
    # Try npm
    if command -v npm &> /dev/null; then
        log_info "Trying npm installation..."
        if npm install -g oh-my-opencode; then
            log_success "Oh My OpenCode installed successfully using npm"
            return 0
        fi
    fi
    
    # Try bun
    if command -v bun &> /dev/null; then
        log_info "Trying bun installation..."
        if bun install -g oh-my-opencode; then
            log_success "Oh My OpenCode installed successfully using bun"
            return 0
        fi
    fi
    
    # Try yarn
    if command -v yarn &> /dev/null; then
        log_info "Trying yarn installation..."
        if yarn global add oh-my-opencode; then
            log_success "Oh My OpenCode installed successfully using yarn"
            return 0
        fi
    fi
    
    # Try pnpm
    if command -v pnpm &> /dev/null; then
        log_info "Trying pnpm installation..."
        if pnpm add -g oh-my-opencode; then
            log_success "Oh My OpenCode installed successfully using pnpm"
            return 0
        fi
    fi
    
    log_error "Failed to install Oh My OpenCode. No suitable package manager found."
    return 1
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    if ! command -v opencode &> /dev/null; then
        log_error "OpenCode is not installed"
        return 1
    fi
    
    if ! command -v oh-my-opencode &> /dev/null; then
        log_error "Oh My OpenCode is not installed"
        return 1
    fi
    
    log_success "Installation verification complete"
    return 0
}

# Show next steps
show_next_steps() {
    log_info "Installation complete! Next steps:"
    echo ""
    echo "1. Authenticate your AI providers:"
    echo "   - Run 'oh-my-opencode' to configure Claude, ChatGPT, or Gemini subscriptions"
    echo "   - Follow the authentication prompts"
    echo ""
    echo "2. Test the installation:"
    echo "   - Run 'opencode --help' to see available commands"
    echo "   - Run 'oh-my-opencode --help' for Oh My OpenCode options"
    echo ""
    echo "3. For more information:"
    echo "   - Visit https://ohmyopencode.com/installation/"
    echo "   - Check the official GitHub repository: https://github.com/code-yeongyu/oh-my-opencode"
    echo ""
    echo "${YELLOW}Important:${NC}"
    echo "   - If you're on OpenCode 1.0.132 or older, an OpenCode bug may break config"
    echo "   - Use a newer version for the best experience"
    echo ""
    log_success "You're ready to use Oh My OpenCode!"
}

# Main execution
main() {
    echo ""
    echo "======================================="
    echo "Oh My OpenCode Full Installation Script"
    echo "======================================="
    echo ""
    
    check_root
    check_dependencies
    
    echo ""
    echo "This script will install:"
    echo "  1. OpenCode (if not already installed)"
    echo "  2. Oh My OpenCode (using the recommended method)"
    echo ""
    echo "Do you want to continue? (y/n)"
    read -r response
    
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Installation cancelled by user."
        exit 0
    fi
    
    echo ""
    install_opencode
    install_ohmyopencode
    verify_installation
    show_next_steps
    
    echo ""
    log_success "Installation completed successfully!"
}

# Run main function
main "$@"