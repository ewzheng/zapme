#!/usr/bin/env bash
# Render and install the zapme systemd service.
#
# Substitutes __USER__ in scripts/zapme.service with $SUDO_USER (the
# real login behind the sudo invocation), writes the rendered unit to
# /etc/systemd/system/zapme.service, reloads systemd, and enables +
# starts the service.
#
# Run from the repo root on the Pi:
#
#     sudo bash scripts/install_systemd.sh
#
# Re-run any time you change scripts/zapme.service. Idempotent — it
# overwrites the installed unit and restarts the service if running.

set -euo pipefail

SERVICE_NAME="zapme.service"
SOURCE_FILE="scripts/${SERVICE_NAME}"
TARGET_FILE="/etc/systemd/system/${SERVICE_NAME}"

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run with sudo (writes to /etc/systemd/system)." >&2
    exit 1
fi

if [[ ! -f "${SOURCE_FILE}" ]]; then
    echo "Cannot find ${SOURCE_FILE}. Run from the repo root." >&2
    exit 1
fi

# $SUDO_USER is the original login that invoked sudo; falls back to
# $USER (which is "root" under sudo) only if SUDO_USER is unset.
TARGET_USER="${SUDO_USER:-${USER}}"
if [[ "${TARGET_USER}" == "root" ]]; then
    echo "Refusing to install with User=root. Run as a normal user via sudo." >&2
    exit 1
fi

# Sanity-check the user exists and has access to the gpio group.
if ! id "${TARGET_USER}" >/dev/null 2>&1; then
    echo "User '${TARGET_USER}' does not exist on this system." >&2
    exit 1
fi
if ! id -nG "${TARGET_USER}" | tr ' ' '\n' | grep -qx gpio; then
    echo "Warning: user '${TARGET_USER}' is not in the 'gpio' group." >&2
    echo "         The service will fail to open /dev/gpiochip*. Fix with:" >&2
    echo "             sudo usermod -aG gpio ${TARGET_USER}" >&2
    echo "         Then log out / back in (or reboot) before starting the service." >&2
fi

echo "Installing ${SERVICE_NAME} for User=${TARGET_USER} -> ${TARGET_FILE}"
sed "s|__USER__|${TARGET_USER}|g" "${SOURCE_FILE}" > "${TARGET_FILE}"
chmod 644 "${TARGET_FILE}"

# Enable user lingering so /run/user/<UID> (the PulseAudio/PipeWire
# socket dir) exists at boot before anyone logs in. Without this,
# audio playback under the system service silently no-ops at boot
# even though the rest of the runtime works.
echo "Enabling user lingering for ${TARGET_USER} (so audio works at boot)..."
loginctl enable-linger "${TARGET_USER}"

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"

if systemctl is-active --quiet "${SERVICE_NAME}"; then
    echo "Restarting already-active ${SERVICE_NAME}..."
    systemctl restart "${SERVICE_NAME}"
else
    echo "Starting ${SERVICE_NAME}..."
    systemctl start "${SERVICE_NAME}"
fi

echo
systemctl status "${SERVICE_NAME}" --no-pager --lines=10 || true
echo
echo "Done. Live logs:  journalctl -u ${SERVICE_NAME} -f"
