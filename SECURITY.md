# Security Policy

## Supported Versions

The following versions of PolyglotLink are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of PolyglotLink seriously. If you discover a security vulnerability, please follow these steps:

### Do NOT

- Open a public GitHub issue for security vulnerabilities
- Disclose the vulnerability publicly before it has been addressed
- Exploit the vulnerability beyond what is necessary to demonstrate the issue

### Do

1. **Email us directly** at security@polyglotlink.io (or open a private security advisory on GitHub)
2. **Include the following information:**
   - Type of vulnerability (e.g., injection, authentication bypass, etc.)
   - Location of the affected code (file path, function name)
   - Step-by-step instructions to reproduce the issue
   - Proof-of-concept or exploit code (if available)
   - Potential impact of the vulnerability
   - Any suggested fixes or mitigations

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 5 business days
- **Updates**: We will keep you informed of our progress toward a fix
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days
- **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices for Users

When deploying PolyglotLink, please follow these security recommendations:

### Authentication & Authorization

- Always use strong, unique passwords for MQTT brokers and databases
- Enable TLS/SSL for all protocol connections
- Rotate API keys and credentials regularly
- Use environment variables for sensitive configuration (never commit secrets)

### Network Security

- Deploy PolyglotLink behind a firewall
- Use private networks for inter-service communication
- Limit exposed ports to only what is necessary
- Enable rate limiting on public endpoints

### LLM Security

- Set appropriate token limits to prevent abuse
- Monitor API usage for anomalies
- Use the minimum necessary model permissions
- Implement input validation before LLM processing

### Data Protection

- Enable encryption at rest for databases
- Use TLS for data in transit
- Implement appropriate access controls
- Regularly backup critical data
- Follow data retention policies

### Container Security

- Use the official Docker images
- Keep base images updated
- Run containers as non-root users (default in our images)
- Scan images for vulnerabilities

### Monitoring & Logging

- Enable audit logging
- Monitor for suspicious activity
- Set up alerts for security events
- Regularly review access logs

## Security Features

PolyglotLink includes the following security features:

- **Input Validation**: All incoming payloads are validated before processing
- **Rate Limiting**: Configurable rate limits on API endpoints
- **TLS Support**: Native support for encrypted connections across protocols
- **Non-root Container**: Docker images run as non-privileged users
- **Secrets Management**: Support for environment-based secrets configuration
- **Security Scanning**: Automated security scanning in CI/CD pipeline (Bandit, Safety, pip-audit)

## Dependency Security

We actively monitor and update dependencies for security vulnerabilities:

- Automated dependency scanning with Dependabot
- Regular security audits with `safety` and `pip-audit`
- Pre-commit hooks to prevent committing secrets

## Security Updates

Security updates are released as patch versions (e.g., 0.1.1, 0.1.2) and announced through:

- GitHub Security Advisories
- Release notes in CHANGELOG.md
- GitHub Releases

We recommend always running the latest patch version of your major.minor release.

## Acknowledgments

We would like to thank the following individuals for responsibly disclosing security issues:

*No security issues have been reported yet.*

---

Thank you for helping keep PolyglotLink and its users safe!
