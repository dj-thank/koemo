# Security policy

## Supported versions

Security fixes are applied to the latest published release line.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue before the
maintainer has had a reasonable opportunity to assess it. Contact the repository
owner through GitHub using a private security advisory when that feature is
available. Include:

- the affected version and commit;
- a minimal reproduction;
- the expected and observed behavior;
- impact and realistic attack preconditions;
- proposed remediation, when known.

This project processes local audio and can optionally call a local Ollama API.
The default implementation rejects non-loopback Ollama endpoints and cloud-routed
model names. Reports that demonstrate a bypass of these controls are especially
valuable.

Do not include real learner recordings, credentials, access tokens, or other
personal data in a report.
