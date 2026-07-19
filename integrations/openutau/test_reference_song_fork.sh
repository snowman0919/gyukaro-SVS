#!/bin/sh
set -eu
root=${1:?usage: test_reference_song_fork.sh /path/to/patched/OpenUtau [dotnet]}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
dotnet="$($here/resolve_dotnet.sh "${2:-dotnet}")"
HOME=${HOME:-"$root/.openutau-home"}
export HOME
mkdir -p "$HOME/.local/share/OpenUtau" "$HOME/.cache/OpenUtau"
: "${GYU_REFERENCE_PROJECT:?set GYU_REFERENCE_PROJECT}"
: "${GYU_REFERENCE_OUTPUT:?set GYU_REFERENCE_OUTPUT}"
: "${GYU_REFERENCE_METRICS:?set GYU_REFERENCE_METRICS}"
cp "$here/OpenUtau.Test/GyuSingerReferenceSongIntegrationTest.cs" "$root/OpenUtau.Test/GyuSingerReferenceSongIntegrationTest.cs"
"$dotnet" test "$root/OpenUtau.Test/OpenUtau.Test.csproj" -c Release --no-restore \
  --filter FullyQualifiedName~GyuSingerReferenceSongIntegrationTest
