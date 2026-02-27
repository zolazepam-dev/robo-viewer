#!/usr/bin/env bash

# Bash completion for Robo-Viewer scripts

# Completion for build_and_run.sh
_robo_viewer_build_and_run_completions() {
    local cur=${COMP_WORDS[COMP_CWORD]}
    local prev=${COMP_WORDS[COMP_CWORD-1]}
    local opts="-h --help -c --clean -b --build-only -r --run-only -v --viewer --verbose --debug --no-optimizations"
    
    COMPREPLY=($(compgen -W "$opts" -- "$cur"))
}

# Completion for train_parallel.sh
_robo_viewer_train_parallel_completions() {
    local cur=${COMP_WORDS[COMP_CWORD]}
    local prev=${COMP_WORDS[COMP_CWORD-1]}
    
    if [ $COMP_CWORD -eq 1 ]; then
        local commands="clean build run all"
        COMPREPLY=($(compgen -W "$commands" -- "$cur"))
    fi
}

# Register completions
complete -F _robo_viewer_build_and_run_completions build_and_run.sh
complete -F _robo_viewer_train_parallel_completions train_parallel.sh