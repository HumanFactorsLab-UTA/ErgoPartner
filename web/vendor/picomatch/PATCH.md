# Local security patch

This directory contains picomatch 4.0.7 from the npm registry with a POSTURE
security patch for GHSA-fc5q-39xc-vjpf. The local version is 4.0.8-posture.1.
This version identifies the local fork. Upstream has no corresponding release.

The parser treats repeated extglobs containing an asterisk as literal text,
using its existing fallback for risky expressions. This includes the reported
single-branch patterns and applies when maxExtglobRecursion is disabled.
Ordinary stars, globstars, braces, character classes and finite extglobs retain
their normal matching behavior.

SECURITY.patch records the parser change. Package metadata changes identify the
fork as private and omit the upstream development scripts and dependencies.
The original MIT license and upstream attribution are retained.

The root web package overrides all picomatch resolution to this directory.
Installation runs scripts/check-picomatch.cjs before the browser download.
The check verifies the actual module used by Vite, fdir and tinyglobby, exercises
ordinary glob matches and runs attack patterns in processes with time limits.

Review this override when upstream publishes a fix and test the replacement
before removing the local package.

[Upstream source](https://github.com/micromatch/picomatch)
[Security advisory](https://github.com/advisories/GHSA-fc5q-39xc-vjpf)
