#!/bin/bash
set -e

DIR="/home/appuser/"
if [ "$(id -u)" = "0" ]; then
    mkdir -p "$DIR"
    chown -R appuser:appuser "$DIR"
    exec gosu appuser "$@"
else
    mkdir -p "$DIR"
    exec "$@"
fi
