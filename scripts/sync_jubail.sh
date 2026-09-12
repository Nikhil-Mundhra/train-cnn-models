#!/bin/bash
# Helper utility to sync data and checkpoints between local machine and NYUAD Jubail HPC

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$REPO_DIR/.env" ]; then
    set -a
    source "$REPO_DIR/.env"
    set +a
fi

NETID="${NETID:-nm4358}"
HPC_HOST="${HPC_HOST:-jubail.abudhabi.nyu.edu}"
SCRATCH_DIR="${HPC_SCRATCH:-/scratch/${NETID}}"
LOCAL_DATA="${LOCAL_BOX_DATASET:-/Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified}"

ACTION="$1"

case "$ACTION" in
    push-data)
        echo "=== Pushing dataset from local Box to Jubail scratch ==="
        echo "Local source:  ${LOCAL_DATA}/"
        echo "Remote dest:   ${NETID}@${HPC_HOST}:${SCRATCH_DIR}/deidentified/"
        rsync -avP "${LOCAL_DATA}/" "${NETID}@${HPC_HOST}:${SCRATCH_DIR}/deidentified/"
        ;;
    pull-checkpoints)
        echo "=== Pulling checkpoints from Jubail scratch to local machine ==="
        mkdir -p "${REPO_DIR}/checkpoints/jubail"
        rsync -avP "${NETID}@${HPC_HOST}:${SCRATCH_DIR}/checkpoints/" "${REPO_DIR}/checkpoints/jubail/"
        ;;
    ssh)
        echo "=== Connecting to Jubail HPC ==="
        ssh "${NETID}@${HPC_HOST}"
        ;;
    *)
        echo "Usage: $0 {push-data|pull-checkpoints|ssh}"
        exit 1
        ;;
esac
