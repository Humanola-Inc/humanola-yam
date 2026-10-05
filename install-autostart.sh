#!/usr/bin/env bash
#
# Usage: ./install-autostart.sh <env_name> <can_ifaces>
#   e.g. ./install-autostart.sh humanola can0,can1
#
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <env_name> <can_ifaces>   (e.g. $0 humanola can0,can1)"
    exit 1
fi

ENV_NAME="$1"
IFS=',' read -r -a CAN_IFACES <<< "$2"

SERVICE_NAME="humanola-yam"
CAN_BITRATE="1000000"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prerequisites
CONDA=""
for candidate in mamba micromamba conda; do
    if command -v "$candidate" >/dev/null 2>&1; then
        CONDA="$candidate"
        break
    fi
done
if [[ -z "$CONDA" ]]; then
    echo "conda/mamba/miniconda not found in PATH"
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found in PATH"
    exit 1
fi
IP_BIN="$(command -v ip)"

# Conda env
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "conda env '$ENV_NAME' already exists, reusing it"
else
    echo "creating conda env '$ENV_NAME'"
    "$CONDA" create -y -n "$ENV_NAME" -c conda-forge python=3.11
fi

echo "installing pinocchio and casadi"
"$CONDA" install -y -n "$ENV_NAME" -c conda-forge pinocchio casadi mujoco

echo "installing libs/i2rt"
cd libs/i2rt
conda run -n "$ENV_NAME" uv pip install -e .
cd ../..

echo "installing libs/i2rt"
conda run -n "$ENV_NAME" uv pip install -e "$REPO_DIR/libs/i2rt"

echo "installing humanola"
conda run -n "$ENV_NAME" uv pip install --extra-index-url https://releases.humanola.com/py/ humanola

# CAN
for iface in "${CAN_IFACES[@]}"; do
    echo "enabling $iface"
    sudo "$IP_BIN" link set "$iface" down || true
    sudo "$IP_BIN" link set "$iface" up type can bitrate "$CAN_BITRATE"
done

# Service (CAN config doesn't survive reboot, so re-apply it before each start)
CAN_PRE=""
for iface in "${CAN_IFACES[@]}"; do
    CAN_PRE+="ExecStartPre=-$IP_BIN link set $iface down"$'\n'
    CAN_PRE+="ExecStartPre=$IP_BIN link set $iface up type can bitrate $CAN_BITRATE"$'\n'
done

ENV_PYTHON=$("$CONDA" run -n "$ENV_NAME" which python3)

echo "installing $SERVICE_NAME service"
sudo tee "/etc/systemd/system/$SERVICE_NAME.service" >/dev/null <<EOF
[Unit]
Description=Humanola YAM
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$REPO_DIR/src
Environment=PYTHONUNBUFFERED=1
${CAN_PRE}ExecStart=$ENV_PYTHON $REPO_DIR/src/main.py
KillSignal=SIGINT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo "done. logs: journalctl -u $SERVICE_NAME -f"
