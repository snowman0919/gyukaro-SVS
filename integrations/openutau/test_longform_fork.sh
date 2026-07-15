#!/bin/sh
set -eu
root=${1:?usage: test_longform_fork.sh /path/to/patched/OpenUtau [dotnet]}
dotnet=${2:-dotnet}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${GYU_RENDERER_URL:?set GYU_RENDERER_URL to the running resident renderer}"
: "${GYU_LONGFORM_OUTPUT:?set GYU_LONGFORM_OUTPUT}"
: "${GYU_LONGFORM_METRICS:?set GYU_LONGFORM_METRICS}"
cp "$here/OpenUtau.Test/GyuSingerLongformIntegrationTest.cs" "$root/OpenUtau.Test/GyuSingerLongformIntegrationTest.cs"
cp "$here/../../examples/openutau_v10_longform.ustx" "$root/OpenUtau.Test/Files/openutau_v10_longform.ustx"
"$dotnet" test "$root/OpenUtau.Test/OpenUtau.Test.csproj" -c Release --filter FullyQualifiedName~GyuSingerLongformIntegrationTest
