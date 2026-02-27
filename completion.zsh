#!/usr/bin/env zsh

# Zsh completion for Robo-Viewer scripts

# Completion for build_and_run.sh
compdef _robo_viewer_build_and_run build_and_run.sh
_robo_viewer_build_and_run() {
    local -a opts
    opts=(-h --help -c --clean -b --build-only -r --run-only -v --viewer --verbose --debug --no-optimizations)
    _describe 'options' opts
}

# Completion for train_parallel.sh
compdef _robo_viewer_train_parallel train_parallel.sh
_robo_viewer_train_parallel() {
    local -a commands
    commands=(clean build run all)
    _describe 'commands' commands
}