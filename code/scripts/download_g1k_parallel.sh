#!/usr/bin/env bash
set -euo pipefail
URL='https://hf-mirror.com/BlinkDL/temp-latest-training-models/resolve/main/rwkv-g1k-1b-temp-6003.pth'
OUT='/hy-tmp/runs/rwkv-g1k-1b-temp-6003.pth'
TMP='/hy-tmp/runs/.rwkv-g1k-1b-temp-6003.parts'
SIZE=3055444605
N=8
mkdir -p "$TMP"
chunk=$(( (SIZE + N - 1) / N ))
pids=()
for ((i=0;i<N;i++)); do
  start=$((i * chunk))
  end=$((start + chunk - 1))
  if (( end >= SIZE )); then end=$((SIZE - 1)); fi
  expected=$((end - start + 1))
  part=$(printf '%s/part.%02d' "$TMP" "$i")
  (
    if [[ -f "$part" ]] && [[ $(stat -c%s "$part") -eq $expected ]]; then exit 0; fi
    rm -f "$part"
    curl -L --fail --retry 12 --retry-delay 2 --retry-all-errors \
      --range "$start-$end" -o "$part" "$URL"
    actual=$(stat -c%s "$part")
    [[ "$actual" -eq "$expected" ]]
  ) >"$TMP/part.$(printf '%02d' "$i").log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
cat "$TMP"/part.?? > "$OUT.tmp"
[[ $(stat -c%s "$OUT.tmp") -eq $SIZE ]]
mv -f "$OUT.tmp" "$OUT"
sha256sum "$OUT" > "$OUT.sha256"
echo COMPLETE size=$(stat -c%s "$OUT")
