#!/bin/bash
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Jump server IP address required" >&2
  exit 1
fi
JUMP_SERVER_IP="$1"

attempts=0
max_attempts=4

while true; do
  attempts=$((attempts + 1))
  set +e
  (
    set -euo pipefail

    MIN_MEM_KB=304857600
    TOTAL_MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    # if [ "$TOTAL_MEM_KB" -ge "$MIN_MEM_KB" ]; then
    #   sudo mkdir -p /home/tmp
    #   sudo rsync -av /home/ubuntu/. /home/tmp/.
    #   sudo mount -t tmpfs -o size=100G tmpfs /home/ubuntu/
    #   sudo chown -R ubuntu:ubuntu /home/ubuntu/
    #   sudo rsync -av /home/tmp/. /home/ubuntu/.
    #   sudo chown -R ubuntu:ubuntu /home/ubuntu/
    # else
      if [ -d "/opt/dlami/nvme" ]; then
        sudo rsync -av /home/ubuntu/. /opt/dlami/nvme/.
        sudo chown -R ubuntu:ubuntu /opt/dlami/nvme
        sudo rm -rf /home/ubuntu/
        sudo ln -s /opt/dlami/nvme /home/ubuntu
      fi
    # fi

    cd /home/ubuntu

    aria2c -x16 -s16 --retry-wait=1 --max-tries=0 --min-split-size=1M --file-allocation=falloc "http://${JUMP_SERVER_IP}/final.squashfs"

    unsquashfs -d yolov9001 -processors "$(nproc)" final.squashfs

    sudo chown -R ubuntu:ubuntu yolov9001
    rm -f final.squashfs

    cd yolov9001
    sudo -u ubuntu bash -l -c "bash ./init.sh"
  )
  exit_code=$?
  set -e
  if [ "${exit_code}" -eq 0 ]; then
    echo "ML server setup complete"
    break
  fi
  if [ "${attempts}" -ge "${max_attempts}" ]; then
    echo "Setup failed after ${max_attempts} attempts" >&2
    exit "${exit_code}"
  fi
  echo "Attempt ${attempts} failed, retrying in 1s..."
  sleep 1
done

cd /home/ubuntu

sudo -u ubuntu screen -dmS ddptrain bash -ilc '
  source ~/.bashrc
  conda activate yolov9
  nvidia-smi >/dev/null 2>&1 || true
  cd ~/yolov9001
  ./yolov9001 ddptrain --optimizer LION --epochs 200 --cache ram
  # ./yolov9001 ddptrain --optimizer SGD --epochs 300 --cache ram --weights ./best.pt --hyp hyp.finetune-coco.yaml
  # exec bash
'