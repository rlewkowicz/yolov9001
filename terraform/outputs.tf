output "mlserver_ids" {
  description = "List of IDs of the EC2 instances in the ML server group."
  value       = module.ml_server_group.ids
}

output "mlserver_name" {
  description = "Name of the EC2 instance group."
  value       = module.ml_server_group.name
}

output "mlserver_public_ips" {
  description = "List of public IP addresses of the EC2 instances in the ML server group."
  value       = module.ml_server_group.public_ips
}

output "ssh_key_name" {
  description = "Name of the SSH key pair."
  value       = aws_key_pair.this.key_name
}

output "jump_id" {
  description = "ID of the Ubuntu EC2 instance."
  value       = module.jump_server.id
}

output "jump_name" {
  description = "Name of the Ubuntu EC2 instance."
  value       = module.jump_server.name
}

output "jump_public_ip" {
  description = "Public IP address of the Ubuntu EC2 instance."
  value       = module.jump_server.public_ip
}