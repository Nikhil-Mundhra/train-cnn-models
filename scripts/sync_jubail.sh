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
LOCAL_NEW_DATA="${LOCAL_BOX_NEW_DATASET:-/Users/nikhilmundhra/Library/CloudStorage/Box-Box/OCT_Segmentations_Solix/deidentified-new}"

ACTION="$1"

RCLONE_BOX="${RCLONE_BOX_REMOTE:-rnfl-box}"
RCLONE_JUBAIL="${RCLONE_JUBAIL_REMOTE:-jubail}"
BOX_CLOUD_PATH="OCT_Segmentations_Solix/deidentified-new"

check_disk() {
    local avail_gb=$(df -g / | awk 'NR==2 {print $4}')
    echo "[Storage] Local Mac available space on /: ${avail_gb} GB"
}

case "$ACTION" in
    push-new-data)
        echo "=== Streaming Disc Cube scans & metadata directly from Box to Jubail (rclone) ==="
        if command -v rclone &> /dev/null; then
            echo "Source (Box API): ${RCLONE_BOX}:${BOX_CLOUD_PATH}"
            echo "Dest (Jubail SFTP): ${RCLONE_JUBAIL}:${SCRATCH_DIR}/deidentified-new"
            echo "Mode: Direct memory streaming (Zero local Mac disk storage used)"
            echo "Filter: Excluding Retina Cube to conserve Jubail inodes & storage"
            rclone copy "${RCLONE_BOX}:${BOX_CLOUD_PATH}" "${RCLONE_JUBAIL}:${SCRATCH_DIR}/deidentified-new" \
                --exclude "*Retina Cube*" \
                --transfers 4 \
                --checkers 8 \
                -P -v
        else
            echo "[Notice] rclone not detected. Falling back to local rsync..."
            check_disk
            rsync -avhP \
                --exclude='*Retina Cube*' \
                --exclude='.DS_Store' \
                "${LOCAL_NEW_DATA}/" \
                "${NETID}@${HPC_HOST}:${SCRATCH_DIR}/deidentified-new/"
        fi
        ;;
    push-full-new-data)
        echo "=== Streaming ENTIRE cohort (Disc Cube + Retina Cube) from Box to Jubail (rclone) ==="
        if command -v rclone &> /dev/null; then
            echo "Source (Box API): ${RCLONE_BOX}:${BOX_CLOUD_PATH}"
            echo "Dest (Jubail SFTP): ${RCLONE_JUBAIL}:${SCRATCH_DIR}/deidentified-new"
            echo "Mode: Direct memory streaming (Zero local Mac disk storage used)"
            rclone copy "${RCLONE_BOX}:${BOX_CLOUD_PATH}" "${RCLONE_JUBAIL}:${SCRATCH_DIR}/deidentified-new" \
                --transfers 4 \
                --checkers 8 \
                -P -v
        else
            echo "[Notice] rclone not detected. Falling back to local rsync..."
            check_disk
            rsync -avhP \
                --exclude='.DS_Store' \
                "${LOCAL_NEW_DATA}/" \
                "${NETID}@${HPC_HOST}:${SCRATCH_DIR}/deidentified-new/"
        fi
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
    clean-legacy)
        echo "=== Purging deprecated legacy cohort on Jubail (/scratch/${NETID}/deidentified) ==="
        ssh "${NETID}@${HPC_HOST}" "rm -rf ${SCRATCH_DIR}/deidentified ${SCRATCH_DIR}/__MACOSX && echo 'Legacy directory deleted successfully.'"
        ;;
    status)
        check_disk
        echo "Jubail Scratch Target: ${SCRATCH_DIR}/deidentified-new/"
        ;;
    *)
        echo "Usage: $0 {push-new-data|push-full-new-data|push-data|pull-checkpoints|pull-reports|clean-legacy|ssh|status}"
        exit 1
        ;;
esac
