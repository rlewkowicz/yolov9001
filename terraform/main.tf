# AWS Specific
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.67"
    }
  }
}

variable "region" {
  type        = string
  description = "AWS region"
  default     = "us-east-1"
}

provider "aws" {
  region = var.region
}

variable "az_letter" {
  default = "b"
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_subnet" "default_in_az" {
  for_each = { for id in data.aws_subnets.default.ids : id => id }
  id       = each.value
}

locals {
  target_az            = "${var.region}${var.az_letter}"
  default_subnet_in_az = try(
    one([
      for s in data.aws_subnet.default_in_az :
      s.id if s.availability_zone == local.target_az
    ]),
    data.aws_subnets.default.ids[0]
  )
}

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) *"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-*-arm64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }

  filter {
    name   = "architecture"
    values = ["arm64"]
  }
}

resource "aws_security_group" "shared" {
  name        = "ml-shared-sg"
  description = "Allow SSH from anywhere and all traffic between group members"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow SSH from anywhere"
  }

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
    description = "Allow all traffic from members of this SG"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }
}

# SSH Keys

variable "ssh_public_key_file" {
  type        = string
  description = "Path to SSH public key file."
  default     = "~/.ssh/id_ed25519.pub"
}

variable "ssh_private_key_file" {
  type        = string
  description = "Path to SSH private key file corresponding to the public key."
  default     = "~/.ssh/id_ed25519"
}

resource "aws_key_pair" "this" {
  key_name   = "ml-key"
  public_key = file(var.ssh_public_key_file)
}

# Prep server and ML servers

module "jump_server" {
  source = "cloudposse/ec2-instance/aws"

  name                        = "jump_server"
  ami                         = data.aws_ami.ubuntu.id
  vpc_id                      = data.aws_vpc.default.id
  subnet                      = local.default_subnet_in_az
  instance_type               = "c8gn.8xlarge"
  associate_public_ip_address = true
  ssh_key_pair                = aws_key_pair.this.key_name
  user_data = <<-EOF
    #!/bin/bash
    set -e
    until sudo apt-get update -y; do echo; done    
    until sudo apt-get install -y nginx zip aria2; do echo; done

    sudo sed -i.bak '$a\
    * soft nofile 200000\n\
    * hard nofile 200000\n\
    * soft nproc unlimited\n\
    * hard nproc unlimited\n\
    * soft core  unlimited\n\
    * hard core  unlimited\n\
    * soft memlock unlimited\n\
    * hard memlock unlimited\n\
    * soft rss unlimited\n\
    * hard rss unlimited\n\
    * soft stack unlimited\n\
    * hard stack unlimited' /etc/security/limits.conf

    sudo tee /etc/sysctl.d/99-highspeed.conf > /dev/null <<EOD
    net.core.somaxconn                 = 32768
    net.ipv4.tcp_max_syn_backlog       = 32768
    net.core.rmem_max                  = 134217728
    net.core.wmem_max                  = 134217728
    net.ipv4.tcp_rmem                  = 4096 87380 134217728
    net.ipv4.tcp_wmem                  = 4096 65536 134217728
    net.ipv4.tcp_window_scaling        = 1
    net.ipv4.tcp_sack                  = 1
    net.ipv4.tcp_congestion_control    = bbr
    net.core.netdev_max_backlog        = 250000
    net.ipv4.tcp_fastopen              = 3
    net.ipv4.tcp_mtu_probing           = 1
    EOD

    sudo sysctl --system
    sudo sysctl -p

    echo 'exec /sbin/reboot -f -p' | sudo tee /sbin/reboot1 > /dev/null
    sudo chmod 777 /sbin/reboot1
    sudo ln -sf /sbin/reboot1 /sbin/poweroff
    sudo ln -sf /sbin/reboot1 /sbin/shutdown
    sudo ln -sf /sbin/reboot1 /sbin/halt

    sudo tee /etc/systemd/system/forcepoweroff.service >/dev/null <<'EOL'
    [Unit]
    Description=Force Immediate Poweroff
    DefaultDependencies=no
    Before=shutdown.target reboot.target halt.target

    [Service]
    Type=oneshot
    ExecStart=/sbin/reboot -f -p
    RemainAfterExit=yes

    [Install]
    WantedBy=halt.target reboot.target shutdown.target
    EOL

    sudo install -d -m755 /etc/systemd/logind.conf.d
    sudo tee    /etc/systemd/logind.conf.d/90-force-power.conf >/dev/null <<'EOD'
    [Login]
    HandlePowerKey=poweroff-force
    PowerKeyIgnoreInhibited=yes
    EOD

    sudo systemctl daemon-reload
    sudo systemctl daemon-reexec
    sudo systemctl enable forcepoweroff.service
    sudo systemctl restart systemd-logind
  EOF

  monitoring           = false
  disable_alarm_action = true
  security_groups      = [aws_security_group.shared.id]
  root_volume_size     = 100
}

module "ml_server_group" {
  source                      = "./terraform-aws-ec2-instance-group"
  ami_owner                   = "amazon"
  region                      = var.region
  name                        = "ml_server"
  ami                         = data.aws_ami.deep_learning.id
  vpc_id                      = data.aws_vpc.default.id
  subnet                      = local.default_subnet_in_az
  security_groups             = [aws_security_group.shared.id]
  associate_public_ip_address = true
  assign_eip_address          = false
  ssh_key_pair                = aws_key_pair.this.key_name 
  generate_ssh_key_pair       = false
  instance_type               = "g6e.48xlarge"
  instance_count              = 1
  root_volume_size            = 70
  monitoring                  = false
  depends_on                  = [null_resource.file_uploader]

  user_data = <<-EOF
    #!/bin/bash
    until sudo apt-get update -y; do echo; done
    until sudo apt-get install -y aria2 screen; do echo; done
    sudo -u ubuntu bash -l -c "curl -LsSf https://astral.sh/uv/install.sh | sh"

    sudo sed -i.bak '$a\
    * soft nofile 200000\n\
    * hard nofile 200000\n\
    * soft nproc unlimited\n\
    * hard nproc unlimited\n\
    * soft core  unlimited\n\
    * hard core  unlimited\n\
    * soft memlock unlimited\n\
    * hard memlock unlimited\n\
    * soft rss unlimited\n\
    * hard rss unlimited\n\
    * soft stack unlimited\n\
    * hard stack unlimited' /etc/security/limits.conf

    sudo tee /etc/sysctl.d/99-highspeed.conf > /dev/null <<EOD
    net.core.somaxconn                 = 32768
    net.ipv4.tcp_max_syn_backlog       = 32768
    net.core.rmem_max                  = 134217728
    net.core.wmem_max                  = 134217728
    net.ipv4.tcp_rmem                  = 4096 87380 134217728
    net.ipv4.tcp_wmem                  = 4096 65536 134217728
    net.ipv4.tcp_window_scaling        = 1
    net.ipv4.tcp_sack                  = 1
    net.ipv4.tcp_congestion_control    = bbr
    net.core.netdev_max_backlog        = 250000
    net.ipv4.tcp_fastopen              = 3
    net.ipv4.tcp_mtu_probing           = 1
    EOD

    sudo sysctl --system
    sudo sysctl -p

    echo 'exec /sbin/reboot -f -p' | sudo tee /sbin/reboot1 > /dev/null
    sudo chmod 777 /sbin/reboot1
    sudo ln -sf /sbin/reboot1 /sbin/poweroff
    sudo ln -sf /sbin/reboot1 /sbin/shutdown
    sudo ln -sf /sbin/reboot1 /sbin/halt

    sudo tee /etc/systemd/system/forcepoweroff.service >/dev/null <<'EOL'
    [Unit]
    Description=Force Immediate Poweroff
    DefaultDependencies=no
    Before=shutdown.target reboot.target halt.target

    [Service]
    Type=oneshot
    ExecStart=/sbin/reboot -f -p
    RemainAfterExit=yes

    [Install]
    WantedBy=halt.target reboot.target shutdown.target
    EOL

    sudo install -d -m755 /etc/systemd/logind.conf.d
    sudo tee    /etc/systemd/logind.conf.d/90-force-power.conf >/dev/null <<'EOD'
    [Login]
    HandlePowerKey=poweroff-force
    PowerKeyIgnoreInhibited=yes
    EOD

    sudo systemctl daemon-reload
    sudo systemctl daemon-reexec
    sudo systemctl enable forcepoweroff.service
    sudo systemctl restart systemd-logind
  EOF

  context = module.this.context
}

# Uploaders and post exec

resource "null_resource" "file_uploader" {
  triggers = {
    instance_id = module.jump_server.id
    script_hash = filesha256("${path.module}/../mktar.py")
  }

  provisioner "local-exec" {
    working_dir = "${path.module}/../"
    command     = "python mktar.py -i last.pt"
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    host        = module.jump_server.public_ip
    private_key = file(var.ssh_private_key_file)
  }

  provisioner "remote-exec" {
    inline = [
      "cloud-init status --wait"
    ]
  }

  provisioner "file" {
    source      = "${path.module}/../archive.tar.gz"
    destination = "/home/ubuntu/archive.tar.gz"
  }

  provisioner "remote-exec" {
    inline = [
      "sudo tar -xzf archive.tar.gz -C /",
      "sudo bash -c 'cd / && bash yolov9001/terraform/prepdeps.sh'"
    ]
  }
}

# File uploaders and system prep

resource "null_resource" "ml_server_wait" {
  count = module.ml_server_group.instance_count

  triggers = {
    instance_id = module.ml_server_group.ids[count.index]
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    host        = module.ml_server_group.public_ips[count.index]
    private_key = file(var.ssh_private_key_file)
    timeout = "300m"
  }

  provisioner "remote-exec" {
    inline = [
      "cloud-init status --wait",
      "echo 'Instance ${self.triggers.instance_id} is ready.'"
    ]
  }
}

resource "null_resource" "ml_server_final_setup" {
  count = module.ml_server_group.instance_count

  depends_on = [null_resource.ml_server_wait]

  triggers = {
    instance_id = module.ml_server_group.ids[count.index]
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    host        = module.ml_server_group.public_ips[count.index]
    private_key = file(var.ssh_private_key_file)
    timeout = "300m"
  }

  provisioner "file" {
    source      = "${path.module}/prepml.sh"
    destination = "/home/ubuntu/prepml.sh"
  }

  provisioner "remote-exec" {
    inline = [
      "sudo tee /ml-hosts.txt > /dev/null <<EOF\n${join("\n", module.ml_server_group.private_ips)}\nEOF",
      "echo 'Created /ml-hosts.txt with all server IPs on instance ${self.triggers.instance_id}.'",
      "sudo bash /home/ubuntu/prepml.sh ${module.jump_server.private_ip}",
    ]
  }
}