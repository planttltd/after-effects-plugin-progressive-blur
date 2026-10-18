#!/bin/zsh
# Builds ProgressiveBlur.plugin (arm64) and runs the GPU/CPU self-check.
# Usage: AE_SDK=/path/to/AfterEffectsSDK/Examples ./build.sh [--install]
set -euo pipefail
cd "${0:A:h}"
: "${AE_SDK:?set AE_SDK to the SDK folder containing Headers/, Util/, Resources/}"
H="$AE_SDK/Headers" U="$AE_SDK/Util" R="$AE_SDK/Resources"
OUT=build/ProgressiveBlur.plugin

# 1) self-check: exact plugin kernels on GPU vs CPU path
mkdir -p build
clang++ -std=c++17 -fobjc-arc -O2 test_fpb.mm -framework Metal -framework Foundation -o build/test_fpb
build/test_fpb

# 2) plugin binary
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources"
clang++ -std=c++17 -fobjc-arc -O3 -arch arm64 -mmacosx-version-min=12.0 -bundle \
  -fvisibility=hidden -I"$H" -I"$H/SP" -I"$U" \
  ProgressiveBlur.mm -framework Metal -framework Foundation \
  -o "$OUT/Contents/MacOS/ProgressiveBlur"

# 3) PiPL resource
xcrun Rez -useDF -d AE_OS_MAC -d __MACH__ -i "$H" -i "$R" ProgressiveBlurPiPL.r \
  -o "$OUT/Contents/Resources/ProgressiveBlur.rsrc"

cat > "$OUT/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>ProgressiveBlur</string>
  <key>CFBundleIdentifier</key><string>com.royplant.ProgressiveBlur</string>
  <key>CFBundleName</key><string>Progressive Blur</string>
  <key>CFBundlePackageType</key><string>eFKT</string>
  <key>CFBundleSignature</key><string>FXTC</string>
  <key>CFBundleVersion</key><string>1.0</string>
</dict></plist>
EOF
printf 'eFKTFXTC' > "$OUT/Contents/PkgInfo"
codesign --force --sign - "$OUT"   # ad-hoc; required to load on Apple Silicon
echo "built $OUT"

if [[ "${1:-}" == "--install" ]]; then
  DEST="/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"
  sudo mkdir -p "$DEST" && sudo rm -rf "$DEST/ProgressiveBlur.plugin" && sudo cp -R "$OUT" "$DEST/"
  echo "installed to $DEST (restart After Effects)"
fi
