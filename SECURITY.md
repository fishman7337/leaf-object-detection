# Security Policy

This project is a research repository, not a production service. The main
security risks are unsafe image inputs, untrusted serialized models, accidental
credential exposure, and unreviewed dependencies.

## Supported Versions

Security fixes apply to the current default branch.

## Reporting

Report concerns privately to the repository maintainer where possible. Include
the affected file, a short impact description, and steps to reproduce. Do not
open a public issue containing secrets, personal data, or an exploit.

## Handling Secrets and Data

- Keep local values outside the repository; never commit credentials or tokens.
- Verify image provenance, consent, and licensing before adding training data.
- Treat model files and uploaded images as untrusted until validated.
- Rotate any secret that was committed or printed in output.

## Runtime Safety

Validate file types and size limits before processing uploads. Review model and
dependency sources, avoid loading untrusted pickle/joblib artifacts, and run the
documented quality and security gates before release.
