#!/bin/sh
set -eu
root=${1:?usage: install_fork.sh /path/to/OpenUtau}
expected=27573ac5c888d927119d5f65a207312d79194b1f
actual=$(git -C "$root" rev-parse HEAD)
if [ "$actual" != "$expected" ]; then
  echo "unsupported OpenUtau revision: $actual (expected $expected)" >&2
  exit 2
fi
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$root/OpenUtau.Core/GyuSinger"
cp "$here/OpenUtau.Core/GyuSinger/GyuSingerRenderer.cs" "$root/OpenUtau.Core/GyuSinger/GyuSingerRenderer.cs"
cp "$here/OpenUtau.Test/GyuSingerRendererTest.cs" "$root/OpenUtau.Test/GyuSingerRendererTest.cs"
cp "$here/../../examples/openutau_v09.ustx" "$root/OpenUtau.Test/Files/openutau_v09.ustx"
git -C "$root" apply --check "$here/openutau-renderers.patch"
git -C "$root" apply "$here/openutau-renderers.patch"
echo "GYU-SINGER renderer installed at OpenUtau $actual"
