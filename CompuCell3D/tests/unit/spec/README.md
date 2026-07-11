# Shared YAML Conformance Draft

This directory contains portable, implementation-neutral CPM unit test cases.

Design rules:

- deterministic local queries only
- no dependency on CC3D class names or method names
- cases should be reusable by other Potts model implementations

Current first-pass runner support in CC3D:

- schema version `1`
- domain `potts_local`
- fixed boundary conditions
- Moore neighborhood declarations
- `volume_energy_delta` queries only
