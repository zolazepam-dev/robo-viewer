#!/bin/bash
# =============================================================================
# JOLTrl Autonomous Agent Dashboard
# Live monitoring + SPS optimization
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Logo
show_logo() {
    echo -e "${CYAN}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    JOLTrl Autonomous Agent Dashboard                         ║"
    echo "║                         Live Monitoring + SPS Optimization                    ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Menu
show_menu() {
    echo -e "${BLUE}┌──────────────────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│${NC}  ${GREEN}1)${NC} Live Agent Monitor (Real-time dashboard)                              │"
    echo -e "${BLUE}│${NC}  ${GREEN}2)${NC} SPS Optimization Agent (Auto-improve performance)                     │"
    echo -e "${BLUE}│${NC}  ${GREEN}3)${NC} Quick SPS Benchmark (Measure current SPS)                             │"
    echo -e "${BLUE}│${NC}  ${GREEN}4)${NC} Build Agent (Classic mode)                                            │"
    echo -e "${BLUE}│${NC}  ${GREEN}5)${NC} View Optimization Report                                              │"
    echo -e "${BLUE}│${NC}  ${GREEN}6)${NC} View Agent Log                                                        │"
    echo -e "${BLUE}│${NC}  ${RED}0)${NC} Exit                                                                    │"
    echo -e "${BLUE}└──────────────────────────────────────────────────────────────────────────────┘${NC}"
    echo
}

# Option 1: Live Agent Monitor
run_live_monitor() {
    echo -e "${CYAN}Starting Live Agent Monitor...${NC}"
    echo -e "${YELLOW}Press 'q' to quit, 'p' to profile SPS${NC}"
    echo
    cd "$PROJECT_ROOT"
    python3 "$SCRIPT_DIR/live_agent_monitor.py"
}

# Option 2: SPS Optimization Agent
run_sps_optimizer() {
    echo -e "${CYAN}Starting SPS Optimization Agent...${NC}"
    echo
    echo -e "${YELLOW}Choose optimization mode:${NC}"
    echo "  1) Scan only (generate report)"
    echo "  2) Auto-apply optimizations (recommended)"
    echo "  3) Custom configuration"
    echo -n "Enter choice [1-3]: "
    read -r mode
    
    cd "$PROJECT_ROOT"
    
    case $mode in
        1)
            echo -e "${CYAN}Running scan-only mode...${NC}"
            python3 "$SCRIPT_DIR/sps_optimizer_agent.py" --scan-only
            ;;
        2)
            echo -e "${CYAN}Running auto-apply optimization...${NC}"
            python3 "$SCRIPT_DIR/sps_optimizer_agent.py" --auto-apply --envs 128 --steps 5000
            ;;
        3)
            echo -n "Number of environments [128]: "
            read -r envs
            envs=${envs:-128}
            
            echo -n "Target SPS [100000]: "
            read -r target
            target=${target:-100000}
            
            echo -n "Auto-apply optimizations? [y/N]: "
            read -r auto
            if [[ "$auto" =~ ^[Yy]$ ]]; then
                python3 "$SCRIPT_DIR/sps_optimizer_agent.py" --auto-apply --envs "$envs" --target-sps "$target"
            else
                python3 "$SCRIPT_DIR/sps_optimizer_agent.py" --envs "$envs" --target-sps "$target"
            fi
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac
}

# Option 3: Quick SPS Benchmark
run_quick_benchmark() {
    echo -e "${CYAN}Running quick SPS benchmark...${NC}"
    echo -n "Number of environments [128]: "
    read -r envs
    envs=${envs:-128}
    
    cd "$PROJECT_ROOT"
    python3 "$SCRIPT_DIR/sps_optimizer_agent.py" --measure-only --envs "$envs" --steps 3000
}

# Option 4: Classic Build Agent
run_build_agent() {
    echo -e "${CYAN}Starting classic autonomous build agent...${NC}"
    cd "$PROJECT_ROOT"
    python3 "$SCRIPT_DIR/autonomous_build_agent.py"
}

# Option 5: View Optimization Report
view_optimization_report() {
    if [[ -f "$PROJECT_ROOT/sps_optimization_report.txt" ]]; then
        echo -e "${CYAN}Opening SPS Optimization Report...${NC}"
        less "$PROJECT_ROOT/sps_optimization_report.txt"
    else
        echo -e "${YELLOW}No optimization report found. Run the SPS optimizer first.${NC}"
    fi
}

# Option 6: View Agent Log
view_agent_log() {
    if [[ -f "$PROJECT_ROOT/agent_log.txt" ]]; then
        echo -e "${CYAN}Opening Agent Log...${NC}"
        tail -100 "$PROJECT_ROOT/agent_log.txt"
    else
        echo -e "${YELLOW}No agent log found.${NC}"
    fi
}

# Main loop
main() {
    clear
    show_logo
    
    while true; do
        show_menu
        echo -n "Enter choice [0-6]: "
        read -r choice
        
        case $choice in
            1)
                run_live_monitor
                ;;
            2)
                run_sps_optimizer
                ;;
            3)
                run_quick_benchmark
                ;;
            4)
                run_build_agent
                ;;
            5)
                view_optimization_report
                ;;
            6)
                view_agent_log
                ;;
            0)
                echo -e "${GREEN}Exiting dashboard. Happy training!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Please try again.${NC}"
                ;;
        esac
        
        echo
        echo -e "${YELLOW}Press Enter to continue...${NC}"
        read -r
        clear
        show_logo
    done
}

# Handle command line args for direct mode
if [[ $# -gt 0 ]]; then
    case $1 in
        --live)
            run_live_monitor
            ;;
        --sps)
            run_sps_optimizer
            ;;
        --benchmark)
            run_quick_benchmark
            ;;
        --build)
            run_build_agent
            ;;
        --report)
            view_optimization_report
            ;;
        --log)
            view_agent_log
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--live|--sps|--benchmark|--build|--report|--log]"
            exit 1
            ;;
    esac
else
    main
fi
