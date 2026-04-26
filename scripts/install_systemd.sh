#!/usr/bin/env bash
# Install the zapme user-level systemd service.
#
# - Removes any prior /etc/systemd/system/zapme.service (system-level
#   leftovers from earlier iterations of this project).
# - Copies scripts/zapme.service to ~/.config/systemd/user/zapme.service
#   (no template substitution — user services are owned by the user
#   that installed them).
# - Enables linger for the user so the user's session (and therefore
#   PipeWire / PulseAudio / this service) starts at boot before
#   anyone has to log in.
# - Enables and starts the user service.
#
# Run from the repo root on the Pi:
#
#     sudo bash scripts/install_systemd.sh
#
# Re-run any time you change scripts/zapme.service. Idempotent.

set -euo pipefail

SERVICE_NAME="zapme.service"
SOURCE_FILE="scripts/${SERVICE_NAME}"
SYSTEM_TARGET="/etc/systemd/system/${SERVICE_NAME}"

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run with sudo (needs to enable-linger and remove any old system unit)." >&2
    exit 1
fi

if [[ ! -f "${SOURCE_FILE}" ]]; then
    echo "Cannot find ${SOURCE_FILE}. Run from the repo root." >&2
    exit 1
fi

# $SUDO_USER is the original login that invoked sudo.
TARGET_USER="${SUDO_USER:-${USER}}"
if [[ "${TARGET_USER}" == "root" ]]; then
    echo "Refusing to install for root. Run as a normal user via sudo." >&2
    exit 1
fi
if ! id "${TARGET_USER}" >/dev/null 2>&1; then
    echo "User '${TARGET_USER}' does not exist on this system." >&2
    exit 1
fi
TARGET_HOME="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"
TARGET_UID="$(id -u "${TARGET_USER}")"
USER_UNIT_DIR="${TARGET_HOME}/.config/systemd/user"
USER_TARGET="${USER_UNIT_DIR}/${SERVICE_NAME}"

# Group-membership warnings (don't auto-fix — group changes need a
# fresh login to take effect, which the script can't trigger).
for grp in gpio audio video; do
    if ! id -nG "${TARGET_USER}" | tr ' ' '\n' | grep -qx "${grp}"; then
        echo "Warning: user '${TARGET_USER}' is not in the '${grp}' group." >&2
        echo "         Fix with: sudo usermod -aG ${grp} ${TARGET_USER}" >&2
        echo "         Then log out / back in (or reboot) before starting the service." >&2
    fi
done

# 1. Tear down any previous system-level install.
if [[ -f "${SYSTEM_TARGET}" ]]; then
    echo "Removing legacy system-level unit at ${SYSTEM_TARGET}..."
    systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
    systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
    rm -f "${SYSTEM_TARGET}"
    systemctl daemon-reload
fi

# 2. Install the user-level unit.
echo "Installing ${SERVICE_NAME} for ${TARGET_USER} -> ${USER_TARGET}"
install -d -o "${TARGET_USER}" -g "${TARGET_USER}" -m 755 "${USER_UNIT_DIR}"
install -o "${TARGET_USER}" -g "${TARGET_USER}" -m 644 \
    "${SOURCE_FILE}" "${USER_TARGET}"

# 3. Enable lingering so the user session (and PipeWire, and this
#    service) starts at boot before anyone has to log in.
echo "Enabling linger for ${TARGET_USER} (so audio + service start at boot)..."
loginctl enable-linger "${TARGET_USER}"

# 4. Reload, enable, (re)start — all in the user's session bus.
#    We use `runuser` to drop into the user's environment, with
#    XDG_RUNTIME_DIR pointing at their /run/user/<UID> dir, so
#    `systemctl --user` can talk to the user's systemd instance.
USER_RUNTIME_DIR="/run/user/${TARGET_UID}"
if [[ ! -d "${USER_RUNTIME_DIR}" ]]; then
    echo "Note: ${USER_RUNTIME_DIR} not present yet (first-time linger);" \
         "user systemd will start on next boot." >&2
fi

run_user() {
    runuser -u "${TARGET_USER}" -- env XDG_RUNTIME_DIR="${USER_RUNTIME_DIR}" "$@"
}

run_user systemctl --user daemon-reload

# Make sure the audio stack actually starts at boot in this user's
# session. Linger gets the user systemd instance running, but
# PipeWire / WirePlumber aren't always enabled by default and won't
# come up automatically without `enable`. Without this, zapme would
# wait on Wants=pipewire.service forever.
echo "Enabling PipeWire / WirePlumber user services for ${TARGET_USER}..."
for svc in pipewire.service pipewire-pulse.service wireplumber.service; do
    if run_user systemctl --user list-unit-files "${svc}" --quiet 2>/dev/null \
        | grep -q "${svc}"; then
        run_user systemctl --user enable "${svc}" || true
    else
        echo "  (skipping ${svc} — not installed)" >&2
    fi
done

run_user systemctl --user enable "${SERVICE_NAME}"

if run_user systemctl --user is-active --quiet "${SERVICE_NAME}"; then
    echo "Restarting already-active ${SERVICE_NAME}..."
    run_user systemctl --user restart "${SERVICE_NAME}"
else
    echo "Starting ${SERVICE_NAME}..."
    run_user systemctl --user start "${SERVICE_NAME}" || {
        echo "Could not start in this session — it will start on next boot via linger." >&2
    }
fi

echo
run_user systemctl --user status "${SERVICE_NAME}" --no-pager --lines=10 || true
echo
echo "Done."
echo "Live logs:  journalctl --user -u ${SERVICE_NAME} -f      (run as ${TARGET_USER})"
echo "or:         sudo journalctl _UID=${TARGET_UID} -u ${SERVICE_NAME} -f"
