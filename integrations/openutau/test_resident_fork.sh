#!/bin/sh
set -eu
root=${1:?usage: test_resident_fork.sh /path/to/patched/OpenUtau [dotnet]}
dotnet=${2:-dotnet}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${GYU_RENDERER_URL:?set GYU_RENDERER_URL to the running resident renderer}"
cp "$here/OpenUtau.Test/GyuSingerResidentIntegrationTest.cs" "$root/OpenUtau.Test/GyuSingerResidentIntegrationTest.cs"
"$dotnet" test "$root/OpenUtau.Test/OpenUtau.Test.csproj" -c Release --filter FullyQualifiedName~GyuSingerResidentIntegrationTest
