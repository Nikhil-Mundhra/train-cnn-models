#!/usr/bin/env python3
"""
Box Drive Cache Manager for macOS Sonoma & Sequoia.
Replaces the deprecated 'fileproviderctl materialize' and 'fileproviderctl evict'.

Usage:
  python3 box_cache.py materialize <path_to_file>
  python3 box_cache.py evict <path_to_file>
  python3 box_cache.py status <path_to_file>
"""

import sys
import os
import subprocess

def get_blocks(path: str) -> int:
    return os.stat(path).st_blocks

def status(path: str) -> None:
    if not os.path.exists(path):
        print(f"Error: File does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    st = os.stat(path)
    is_resident = st.st_blocks > 0
    state = f"RESIDENT ON DISK ({st.st_blocks * 512 / (1024*1024):.2f} MB)" if is_resident else "DATALESS (Online-only placeholder)"
    print(f"File: {os.path.basename(path)}")
    print(f"Path: {path}")
    print(f"Status: {state}")
    print(f"Logical size: {st.st_size / (1024*1024):.2f} MB | Allocated blocks: {st.st_blocks}")

def materialize(path: str) -> None:
    if not os.path.exists(path):
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    if os.path.isdir(path):
        print(f"[BoxCache] Recursively hydrating directory: {path}...")
        total_hydrated = 0
        for root, _, files in os.walk(path):
            for file in files:
                fpath = os.path.join(root, file)
                if not file.startswith('.'):
                    with open(fpath, "rb") as f:
                        while chunk := f.read(4 * 1024 * 1024):
                            pass
                    total_hydrated += 1
        print(f"[BoxCache] Hydrated {total_hydrated} files in {os.path.basename(path)}.")
    else:
        print(f"[BoxCache] Hydrating {os.path.basename(path)}...")
        with open(path, "rb") as f:
            while chunk := f.read(4 * 1024 * 1024):
                pass
        st = os.stat(path)
        print(f"[BoxCache] Hydration complete! Resident blocks: {st.st_blocks} ({st.st_blocks * 512 / (1024*1024):.2f} MB)")

def evict(path: str) -> None:
    if not os.path.exists(path):
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    abs_path = os.path.abspath(path)
    target_type = "directory" if os.path.isdir(abs_path) else "file"
    print(f"[BoxCache] Evicting {target_type}: {os.path.basename(abs_path)}...")
    swift_script = f"""import Foundation
let url = URL(fileURLWithPath: "{abs_path}")
do {{
    try FileManager.default.evictUbiquitousItem(at: url)
    exit(0)
}} catch {{
    fputs("Eviction error: \\(error)\\n", stderr)
    exit(1)
}}
"""
    res = subprocess.run(["swift", "-"], input=swift_script, text=True, capture_output=True)
    if res.returncode == 0:
        st = os.stat(path)
        print(f"[BoxCache] Eviction complete! Reverted to dataless cloud placeholder (Blocks: {st.st_blocks})")
    else:
        print(f"[BoxCache] Eviction failed: {res.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    action = sys.argv[1].lower()
    target = sys.argv[2]
    if action in ["materialize", "hydrate", "download"]:
        materialize(target)
    elif action in ["evict", "purge", "online-only"]:
        evict(target)
    elif action in ["status", "info"]:
        status(target)
    else:
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()
