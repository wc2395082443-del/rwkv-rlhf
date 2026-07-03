#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/helicopter
mkdir -p /root/autodl-tmp/helicopter_archives
fetch_archive() {
  repo="$1"; commit="$2"; dest="$3"; base="${repo##*/}-${commit}.tar.gz"; tarball="/root/autodl-tmp/helicopter_archives/${base}"
  url="https://codeload.github.com/${repo}/tar.gz/${commit}"
  echo "[$(date '+%F %T')] fetch ${repo}@${commit} -> ${tarball}"
  rm -f "${tarball}.tmp"
  for i in 1 2 3 4 5; do
    echo "attempt $i"
    if curl -L --fail --retry 3 --retry-delay 5 --connect-timeout 60 --max-time 1800 -o "${tarball}.tmp" "$url"; then
      mv "${tarball}.tmp" "$tarball"
      break
    fi
    sleep 10
  done
  test -s "$tarball"
  rm -rf "$dest"
  mkdir -p "$dest"
  tar -xzf "$tarball" --strip-components=1 -C "$dest"
  echo "$commit" > "$dest/.helicopter_submodule_commit"
  echo "[$(date '+%F %T')] done ${repo}; size=$(du -sh "$dest" | cut -f1)"
}
fetch_archive rwkv-rs/rwkv-lm 5879a4a00ac96d6da866e1a3da371a9b13cfa2d0 src/train/rwkv-lm
fetch_archive rwkv-rs/verl-rwkv c7721d61122f6e108cb8913be4b9563663cd0dae src/train/verl-rwkv
echo "[$(date '+%F %T')] ALL_DONE"

