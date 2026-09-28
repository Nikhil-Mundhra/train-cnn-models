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
LOCAL_NEW_DATA="${LOCAL_BOX_NEW_DATASET:-/Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified-new}"

ACTION="$1"

check_disk() {
    local avail_gb=$(df -g / | awk 'NR==2 {print $4}')
    echo "[Storage] Local Mac available space on /: ${avail_gb} GB"
    if [ "$avail_gb" -lt 20 ]; then
        echo "  [WARNING] Free disk space is low (<20 GB). Do NOT run full cohort transfer."
    fi
}

case "$ACTION" in
    push-new-data)
        echo "=== Pushing Disc Cube scans & metadata from Box to Jubail ==="
        check_disk
        echo "Local source:  ${LOCAL_NEW_DATA}/"
        echo "Remote dest:   ${NETID}@${HPC_HOST}:${SCRATCH_DIR}/deidentified-new/"
        echo "Filter:        Excluding Retina Cube to prevent local disk exhaustion (saving ~30 GB)"
        rsync -avhP \
            --exclude='*Retina Cube*' \
            --exclude='.DS_Store' \
            "${LOCAL_NEW_DATA}/" \
            "${NETID}@${HPC_HOST}:${SCRATCH_DIR}/deidentified-new/"
        ;;
    push-full-new-data)
        echo "=== Pushing ENTIRE cohort (Disc Cube + Retina Cube) from Box to Jubail ==="
        check_disk
        avail_gb=$(df -g / | awk 'NR==2 {print $4}')
        if [ "$avail_gb" -lt 36 ]; then
            echo "[CAUTION] Full cohort requires ~36 GB cloud download into Box cache, but only ${avail_gb} GB is free!"
            echo "Proceeding may cause disk space exhaustion. Press Ctrl+C within 5s to cancel..."
            sleep 5
        fi
        echo "Local source:  ${LOCAL_NEW_DATA}/"
        echo "Remote dest:   ${NETID}@${HPC_HOST}:${SCRATCH_DIR}/deidentified-new/"
        rsync -avhP \
            --exclude='.DS_Store' \
            "${LOCAL_NEW_DATA}/" \
            "${NETID}@${HPC_HOST}:${SCRATCH_DIR}/deidentified-new/"
        ;;
    push-data)
        echo "=== Pushing legacy dataset from local Box to Jubail scratch ==="
        echo "Local source:  ${LOCAL_DATA}/"
        echo "Remote dest:   ${NETID}@${HPC_HOST}:${SCRATCH_DIR}/deidentified/"
        rsync -avhP --exclude='.DS_Store' "${LOCAL_DATA}/" "${NETID}@${HPC_HOST}:${SCRATCH_DIR}/deidentified/"
        ;;
    pull-checkpoints)
        echo "=== Pulling checkpoints from Jubail scratch to local machine ==="
        mkdir -p "${REPO_DIR}/checkpoints/jubail"
        rsync -avhP "${NETID}@${HPC_HOST}:${SCRATCH_DIR}/checkpoints/" "${REPO_DIR}/checkpoints/jubail/"
        ;;
    pull-reports)
        echo "=== Pulling clinical evaluation reports from Jubail scratch ==="
        mkdir -p "${REPO_DIR}/docs/cohort_reports"
        rsync -avhP "${NETID}@${HPC_HOST}:${SCRATCH_DIR}/reports/" "${REPO_DIR}/docs/cohort_reports/"
        ;;
    ssh)
        echo "=== Connecting to Jubail HPC ==="
        ssh "${NETID}@${HPC_HOST}"
        ;;
    status)
        check_disk
        echo "Jubail Scratch Target: ${SCRATCH_DIR}/deidentified-new/"
        ;;
    *)
        echo "Usage: $0 {push-new-data|push-full-new-data|push-data|pull-checkpoints|pull-reports|ssh|status}"
        exit 1
        ;;
esac
