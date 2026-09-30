# Security

Report a vulnerability, or a leaked secret such as a Visin token in the repository or its history,
privately through GitHub's "Report a vulnerability" (Security tab), not in a public issue.

Secrets never belong in configs or code: Visin tokens go in `.env` (ignored by git) or the
environment, and in GitHub Actions secrets for CI.
