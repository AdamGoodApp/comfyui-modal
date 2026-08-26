#!/usr/bin/env bash
# Push this working tree to the ComfyUI server's custom_nodes and restart.
#
# The server's auto_models.json (live wishlist) and .deployed_version are
# preserved across redeploys: excluded from the tar AND stashed around the
# rm -rf that replaces the directory.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="root@100.75.155.26"
BASE="$(basename "$REPO_DIR")"
DEST=/opt/ComfyUI/custom_nodes/comfyui-modal

tar czf - --exclude .git --exclude .venv --exclude __pycache__ \
    --exclude auto_models.json --exclude .deployed_version \
    -C "$(dirname "$REPO_DIR")" "$BASE" \
  | ssh "$SERVER" "
      set -eu
      STASH=\$(mktemp -d)
      for f in auto_models.json .deployed_version; do
        [ -f $DEST/\$f ] && cp -p $DEST/\$f \$STASH/ || true
      done
      rm -rf $DEST
      tar xzf - -C /opt/ComfyUI/custom_nodes
      [ '$BASE' = comfyui-modal ] || mv /opt/ComfyUI/custom_nodes/$BASE $DEST
      for f in auto_models.json .deployed_version; do
        [ -f \$STASH/\$f ] && cp -p \$STASH/\$f $DEST/ || true
      done
      rm -rf \$STASH
    "
ssh "$SERVER" "systemctl restart comfyui"
echo "Deployed. Logs: ssh $SERVER journalctl -u comfyui -n 50 -f"
