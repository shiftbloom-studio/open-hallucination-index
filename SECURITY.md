# Security Policy

## Supported Versions

We only provide security updates for the latest version of Open Hallucination Index.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability within Open Hallucination Index, please help us by reporting it responsibly.

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, please send an email to **security@shiftbloom.studio** with the following information:

1.  **Description**: A detailed description of the vulnerability.
2.  **Reproduction**: Steps to reproduce the issue (including sample code/requests).
3.  **Impact**: What could an attacker achieve with this vulnerability?
4.  **Versions**: Which versions of OHI are affected?

We will acknowledge your report within 48 hours and provide a timeline for a fix. We follow a 90-day disclosure policy from the time the vulnerability is confirmed.

## Security Practices in OHI

-   **API Key Auth**: Required by default for all data-modifying or costly operations.
-   **Dependency Scanning**: Automated scans via GitHub Actions.
-   **Environment Isolation**: Designed to run in isolated Docker containers.
-   **Secret Management**: Never commit `.env` files; use CI/CD secrets.
