# OpenUtau Packaging Safety

The safe packager refuses RC8 because it is rejected and RC9 because it is blocked. It also refuses every other current backend because none is production-approved or enabled for OpenUtau packaging.

An explicit diagnostic mode exists only to inspect reproducible metadata shape. It requires an output directory ending in `-diagnostic`, disables model activation, bundles no checkpoints or audio, labels the package `NOT A RELEASE`, and records SHA-256 for every metadata file. This mode does not satisfy the reproducible release-manifest gate and is not an installable voicebank release.

No production package, OpenUtau library, archive, tag, or release was created in this phase.
