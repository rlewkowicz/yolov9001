#!/usr/bin/env bash
set -euo pipefail

LABELS_URL="https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"
IMAGES_BASE="http://images.cocodataset.org/zips"
OUT_ROOT="."
LABELS_DIR="${OUT_ROOT}"
IMAGES_DIR="${OUT_ROOT}/coco/images"

JOBS="${JOBS:-4}"
CONNS_PER_FILE="${CONNS_PER_FILE:-16}"
RETRY_WAIT=1

TMP_META_DIR="$(mktemp -d /tmp/coco_zip_meta.XXXXXXXX)"
DONE_DIR="${TMP_META_DIR}/done"
mkdir -p "${DONE_DIR}"
export TMP_META_DIR DONE_DIR

EXPECTED=("coco2017labels-segments" "train2017" "val2017" "test2017")

mkdir -p "${LABELS_DIR}" "${IMAGES_DIR}"

checksum_and_verify() {
    local zip="$1" base="$2" dest="$3"
    local chk="${TMP_META_DIR}/${base}.md5"
    local lst="${TMP_META_DIR}/${base}.lst"
    md5sum "$zip" | awk '{print $1}' >"$chk"
    unzip -Z1 "$zip" | sed 's#^[./]*##' >"$lst"
    mkdir -p "$dest"
    unzip -q -o "$zip" -d "$dest"
    local missing=0
    while IFS= read -r rel; do
        [[ "$rel" == */ ]] && continue
        [ -e "$dest/$rel" ] || { echo "Missing $dest/$rel" >&2; missing=1; }
    done <"$lst"
    [ "$missing" -eq 0 ] || { echo "Verify failed for $zip" >&2; exit 1; }
    rm -f "$zip"
    touch "${DONE_DIR}/${base}.done"
    echo "[OK] $base verified"
}

if command -v aria2c >/dev/null 2>&1; then
    ARIA_LIST="$(mktemp)"
    ARIA_HOOK="$(mktemp)"
    cleanup() { rm -f "$ARIA_LIST" "$ARIA_HOOK"; rm -rf "$TMP_META_DIR"; }
    trap cleanup EXIT
    {
        printf '%s\n  out=%s\n  dir=%s\n' "$LABELS_URL" "$(basename "$LABELS_URL")" "$OUT_ROOT"
        for p in train2017.zip val2017.zip test2017.zip; do
            printf '%s/%s\n  out=%s\n  dir=%s\n' "$IMAGES_BASE" "$p" "$p" "$OUT_ROOT"
        done
    } >"$ARIA_LIST"
    cat >"$ARIA_HOOK" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail
zip_path="$3"
name="$(basename "$zip_path")"
case "$name" in
  coco2017labels-segments.zip) dest="." ;;
  train2017.zip|val2017.zip|test2017.zip) dest="./coco/images" ;;
  *) dest="." ;;
esac
checksum_and_verify() {
    local zip="$1" base="$2" dest="$3"
    local chk="${TMP_META_DIR}/${base}.md5"
    local lst="${TMP_META_DIR}/${base}.lst"
    md5sum "$zip" | awk '{print $1}' >"$chk"
    unzip -Z1 "$zip" | sed 's#^[./]*##' >"$lst"
    mkdir -p "$dest"
    unzip -q -o "$zip" -d "$dest"
    local missing=0
    while IFS= read -r rel; do
        [[ "$rel" == */ ]] && continue
        [ -e "$dest/$rel" ] || missing=1
    done <"$lst"
    [ "$missing" -eq 0 ] || { echo "Verify failed $zip" >&2; exit 1; }
    rm -f "$zip"
    touch "${DONE_DIR}/${base}.done"
    echo "[OK] $base verified"
}
checksum_and_verify "$zip_path" "${name%.zip}" "$dest"
HOOK
    chmod +x "$ARIA_HOOK"
    aria2c \
      --input-file="$ARIA_LIST" \
      --continue=true \
      --allow-overwrite=true \
      --auto-file-renaming=false \
      --max-concurrent-downloads="$JOBS" \
      --max-connection-per-server="$CONNS_PER_FILE" \
      --split="$CONNS_PER_FILE" \
      --min-split-size=1M \
      --retry-wait="$RETRY_WAIT" \
      --max-tries=0 \
      --on-download-complete="$ARIA_HOOK"
    while [ "$(ls "$DONE_DIR" | wc -l)" -lt "${#EXPECTED[@]}" ]; do
        sleep 1
    done
    rm -rf "$TMP_META_DIR"
    echo "All downloads and verified extractions finished."
    exit 0
fi

echo "aria2c not found - using curl fallback."

fetch() {
    curl -L --fail --retry 10 --retry-delay 3 --retry-all-errors \
         --continue-at - -o "$2" "$1"
}

download_bg() {
    local url="$1" zip="$2" dest="$3"
    ( fetch "$url" "$zip"
      checksum_and_verify "$zip" "$(basename "$zip" .zip)" "$dest" ) &
}

fetch "$LABELS_URL" "$OUT_ROOT/$(basename "$LABELS_URL")"
checksum_and_verify "$OUT_ROOT/$(basename "$LABELS_URL")" "coco2017labels-segments" "$OUT_ROOT"

download_bg "$IMAGES_BASE/train2017.zip" "$OUT_ROOT/train2017.zip" "$IMAGES_DIR"
download_bg "$IMAGES_BASE/val2017.zip" "$OUT_ROOT/val2017.zip" "$IMAGES_DIR"
download_bg "$IMAGES_BASE/test2017.zip" "$OUT_ROOT/test2017.zip" "$IMAGES_DIR"

wait
while [ "$(ls "$DONE_DIR" | wc -l)" -lt "${#EXPECTED[@]}" ]; do
    echo "verifying downloads, please wait"
    sleep 1
done
rm -rf "$TMP_META_DIR"
echo "All downloads and verified extractions finished."