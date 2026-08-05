# Security Policy

## Supported Versions

Security fixes are applied to the current `main` branch. The project does not
currently maintain long-lived release branches.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately to the repository maintainer
instead of opening a public issue.

Include:

- affected commit or release
- reproduction steps
- expected and observed behavior
- impact assessment
- any relevant logs with secrets removed

This repository is designed to run locally with deterministic defaults. Do not
commit API keys, private datasets, model credentials, traces containing sensitive
user data, or generated artifacts that include confidential content.
