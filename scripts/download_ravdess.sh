#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${RAVDESS_DIR:-data/RAVDESS_speech}
ZIP=data/Audio_Speech_Actors_01-24.zip
MD5=bc696df654c87fed845eb13823edef8a
URL="https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"

[ -f "$ZIP" ] || curl -fL -o "$ZIP" "$URL"
python3 -c "import hashlib, sys; h = hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest(); sys.exit(f'md5 mismatch: {h}' if h != sys.argv[2] else 0)" "$ZIP" "$MD5"
mkdir -p "$DEST"
unzip -q -n "$ZIP" -d "$DEST"
echo "$(ls "$DEST"/Actor_*/03-01-*.wav | wc -l | tr -d ' ') speech clips in $DEST"
