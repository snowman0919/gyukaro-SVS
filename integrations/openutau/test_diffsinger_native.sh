#!/bin/sh
set -eu
root=${1:?usage: test_diffsinger_native.sh /path/to/OpenUtau [dotnet]}
dotnet=${2:-dotnet}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${GYU_DIFFSINGER_PACKAGE:?set GYU_DIFFSINGER_PACKAGE}"
: "${GYU_DIFFSINGER_UST:?set GYU_DIFFSINGER_UST}"
: "${GYU_DIFFSINGER_OUTPUT:?set GYU_DIFFSINGER_OUTPUT}"
: "${GYU_DIFFSINGER_METRICS:?set GYU_DIFFSINGER_METRICS}"
: "${XDG_DATA_HOME:?set XDG_DATA_HOME}"
: "${XDG_CACHE_HOME:?set XDG_CACHE_HOME}"
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
