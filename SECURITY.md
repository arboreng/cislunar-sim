# Security Policy

## Reporting a Vulnerability

Please do not file a public GitHub issue for security vulnerabilities.

Report security issues by email to **sean@arboreng.com**. Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce or proof-of-concept code
- Any suggested mitigations

You will receive an acknowledgement within 48 hours and a status update within 7 days.

## Supported Versions

Security fixes are provided for the latest released minor version.

| Version | Supported |
| --- | --- |
| 1.0.x | Yes |

## Scope

cislunar-sim is a pure-Python simulation library with no persistent server
process, no authentication layer, and no inbound network surface. The main
security-relevant areas are:

- **Dependency vulnerabilities** in numpy or astropy
- **Malicious or malformed local input data** passed to telemetry loaders such as `load_satnogs_frames()` and `load_historical_tle_series()`
- **Optional outbound HTTP fetch helpers** (for example NOAA and CelesTrak clients) that download untrusted remote data and cache it locally

Pure numerical propagation, guidance-law calculations, and deterministic
validation utilities are not network services and are generally out of scope
unless a report demonstrates a concrete safety or integrity impact.

## Disclosure Policy

We follow coordinated disclosure. Please allow 90 days for a fix before public disclosure. We will credit reporters in the release notes unless you prefer to remain anonymous.
