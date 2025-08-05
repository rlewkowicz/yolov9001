#!/bin/bash
set -e

cd yolov9001
bash scripts/get_coco.sh
cd /

sudo mksquashfs /yolov9001 /var/www/html/final.squashfs -comp zstd -Xcompression-level 1 -b 4K
sudo chown -R www-data:www-data /var/www/html
sudo cp yolov9001/terraform/nginx.conf /etc/nginx/nginx.conf
sudo service nginx reload