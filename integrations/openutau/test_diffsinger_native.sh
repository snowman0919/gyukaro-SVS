#!/bin/sh
set -eu
root=${1:?usage: test_diffsinger_native.sh /path/to/OpenUtau [dotnet]}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
dotnet="$($here/resolve_dotnet.sh "${2:-dotnet}")"
HOME=${HOME:-"$root/.openutau-home"}
export HOME
export XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
mkdir -p "$HOME/.local/share/OpenUtau" "$HOME/.cache/OpenUtau"
: "${GYU_DIFFSINGER_PACKAGE:?set GYU_DIFFSINGER_PACKAGE}"
: "${GYU_DIFFSINGER_UST:?set GYU_DIFFSINGER_UST}"
: "${GYU_DIFFSINGER_OUTPUT:?set GYU_DIFFSINGER_OUTPUT}"
: "${GYU_DIFFSINGER_METRICS:?set GYU_DIFFSINGER_METRICS}"
singer_id=${GYU_DIFFSINGER_SINGER_ID:-GYU-DiffSinger-JA-eval}
singers="$XDG_DATA_HOME/OpenUtau/Singers"
rm -rf "$singers/$singer_id"
mkdir -p "$singers"
cp -R "$GYU_DIFFSINGER_PACKAGE" "$singers/$singer_id"
cp "$here/OpenUtau.Test/DiffSingerNativeIntegrationTest.cs" \
  "$root/OpenUtau.Test/DiffSingerNativeIntegrationTest.cs"
export GYU_DIFFSINGER_SINGER_ID=$singer_id
"$dotnet" test "$root/OpenUtau.Test/OpenUtau.Test.csproj" -c Release --no-restore \
  --filter FullyQualifiedName~DiffSingerNativeIntegrationTest
