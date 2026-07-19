#!/bin/sh
set -eu
root=${1:?usage: test_longform_fork.sh /path/to/patched/OpenUtau [dotnet]}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
dotnet="$($here/resolve_dotnet.sh "${2:-dotnet}")"
HOME=${HOME:-"$root/.openutau-home"}
export HOME
mkdir -p "$HOME/.local/share/OpenUtau" "$HOME/.cache/OpenUtau"
: "${GYU_RENDERER_URL:?set GYU_RENDERER_URL to the running resident renderer}"
: "${GYU_LONGFORM_OUTPUT:?set GYU_LONGFORM_OUTPUT}"
: "${GYU_LONGFORM_METRICS:?set GYU_LONGFORM_METRICS}"

export XDG_DATA_HOME=${XDG_DATA_HOME:-"$root/.openutau-data"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$root/.openutau-cache"}
rm -rf "$XDG_DATA_HOME/OpenUtau" "$XDG_CACHE_HOME/OpenUtau"

cp "$here/OpenUtau.Test/GyuSingerLongformIntegrationTest.cs" "$root/OpenUtau.Test/GyuSingerLongformIntegrationTest.cs"
cp "$here/../../examples/openutau_v10_longform.ustx" "$root/OpenUtau.Test/Files/openutau_v10_longform.ustx"
"$dotnet" test "$root/OpenUtau.Test/OpenUtau.Test.csproj" -c Release --filter FullyQualifiedName~GyuSingerLongformIntegrationTest
