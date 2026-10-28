# Security Policy

## Secret isolation

Cloud credentials belong on the host only. Jetson / Linux robot clients should not receive host API keys.

Do not commit:

- real `.env` files;
- API keys or access tokens;
- TLS private keys;
- speaker-profile embeddings or private voice recordings;
- robot deployment credentials.

If a credential is exposed, revoke or rotate it first, then remove the tracked file. Git history should still be treated as exposed.

## Robot command boundary

The hub does not treat LLM output as trusted robot control.

Robot-specific adapters and device-side bridges must validate commands before execution. Real motion remains disabled by default and should require an explicit local permission gate.

The robot controller remains responsible for physical safety, including workspace, joint, velocity, collision, force and emergency-stop constraints.

## Network exposure

The raw voice / robot WebSocket service is designed for trusted LAN or private-network use. Do not expose robot-control endpoints directly to the public Internet without an authenticated transport and a reviewed deployment architecture.

## Reports

Please report validation bypasses, credential exposure, unsafe execution paths or protocol-security issues through a GitHub issue containing only the minimum non-sensitive reproduction information.
