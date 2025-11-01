"""
AWS EC2 Automation Tutorial
---------------------------
This script demonstrates how to:
1. Create a key pair for secure SSH connection
2. Launch an EC2 instance
3. Create and configure a security group
4. Attach the security group to an existing instance
5. Start, stop, and terminate EC2 instances programmatically
"""

import boto3
import time
import os

# ============================================================
# 1️⃣  INITIALIZE EC2 CLIENT
# ============================================================
# Create a boto3 EC2 client (ensure 'aws configure' is done beforehand)
# Set region according to your AWS setup, e.g., ap-southeast-2 for Sydney
ec2 = boto3.client('ec2', region_name='ap-southeast-2')

# Describe all current instances to verify connection works
print("Checking existing EC2 instances...")
print(ec2.describe_instances())

# ============================================================
# 2️⃣  CREATE A KEY PAIR FOR SSH ACCESS
# ============================================================
# A key pair allows you to securely connect to your instance via SSH.
# The private key (.pem) is saved locally; AWS stores only the public key.
# DO NOT upload this file to GitHub or share it publicly.
print("\nCreating new key pair: TestLinux")

response = ec2.create_key_pair(KeyName='TestLinux')

# Create a directory to store credentials if it doesn't exist
os.makedirs("Cred", exist_ok=True)

# Save the private key to a local PEM file
with open("Cred/TestLinux.pem", "w") as file:
    file.write(response['KeyMaterial'])

print("Key pair created and saved to Cred/TestLinux.pem")
print("Remember to set permissions: chmod 400 Cred/TestLinux.pem\n")

# ============================================================
# 3️⃣  CREATE A SECURITY GROUP
# ============================================================
# Security groups act as virtual firewalls for your EC2 instance.
# Here we create one called "TestLinux" that allows SSH (port 22).
# ⚠️ 0.0.0.0/0 allows access from anywhere; restrict it in production.
print("Creating security group...")

response = ec2.create_security_group(
    GroupName="TestLinux",
    Description="Security group for testing..."
)

security_group_id = response['GroupId']
print(f"Created Security Group: {security_group_id}")

# Add inbound rule for SSH (port 22)
ec2.authorize_security_group_ingress(
    GroupId=security_group_id,
    IpPermissions=[
        {
            "IpProtocol": "tcp",
            "FromPort": 22,
            "ToPort": 22,
            "IpRanges": [{"CidrIp": "0.0.0.0/0"}]
        }
    ]
)
print("Inbound SSH rule added successfully.\n")

# ============================================================
# 4️⃣  LAUNCH AN EC2 INSTANCE
# ============================================================
# Now launch an Ubuntu EC2 instance using the key pair and security group.
# ImageId: Replace with the correct Ubuntu AMI for your region.
# InstanceType: Choose appropriate size (e.g., t3.small, t2.micro, etc.)
print("Launching EC2 instance...")

response = ec2.run_instances(
    ImageId='ami-0bdd88bd06d16ba03',  # Ubuntu 22.04 (ap-southeast-2)
    InstanceType='t3.small',
    MinCount=1,
    MaxCount=1,
    KeyName='TestLinux',
    BlockDeviceMappings=[
        {
            "DeviceName": "/dev/xvda",
            "Ebs": {
                "DeleteOnTermination": True,
                "VolumeSize": 20
            }
        }
    ]
)

# Extract instance ID from response
Instance_Id = response['Instances'][0]['InstanceId']
print(f"Instance launched successfully with ID: {Instance_Id}\n")

# ============================================================
# 5️⃣  ATTACH SECURITY GROUP TO INSTANCE
# ============================================================
# This step ensures our new security group is applied to the running instance.
# Each instance must have at least one security group; we append ours to it.
print("Attaching security group to instance...")

res = ec2.describe_instances(InstanceIds=[Instance_Id])
OldGroupId = res['Reservations'][0]['Instances'][0]['SecurityGroups'][0]['GroupId']

ec2.modify_instance_attribute(
    InstanceId=Instance_Id,
    Groups=[OldGroupId, security_group_id]
)

print(f"Security group {security_group_id} attached to instance {Instance_Id}\n")

# ============================================================
# 6️⃣  INSTANCE MANAGEMENT FUNCTIONS
# ============================================================
# Define helper functions to start, stop, and terminate the instance.
# The wait_for_status() function polls the instance status until
# the target state (running/stopped/terminated) is reached.
def wait_for_status(instance_id, target_status):
    while True:
        response = ec2.describe_instances(InstanceIds=instance_id)
        status = response['Reservations'][0]['Instances'][0]['State']['Name']
        if status == target_status:
            print(f"✅ Instance is now in '{target_status}' state.")
            break
        print(f"⏳ Current status: {status} ... waiting to reach {target_status}")
        time.sleep(10)

def start_instances(instance_id):
    print("▶️ Starting EC2 instance...")
    ec2.start_instances(InstanceIds=instance_id)
    wait_for_status(instance_id, 'running')

def stop_instances(instance_id):
    print("⏹ Stopping EC2 instance...")
    ec2.stop_instances(InstanceIds=instance_id)
    wait_for_status(instance_id, 'stopped')

def terminate_instances(instance_id):
    print("💀 Terminating EC2 instance...")
    ec2.terminate_instances(InstanceIds=instance_id)
    wait_for_status(instance_id, 'terminated')

# ============================================================
# 7️⃣  CONTROL INSTANCE LIFECYCLE
# ============================================================
# You can comment out the steps you don’t want to execute.
# This section demonstrates how to start, stop, and terminate the instance.

start_instances([Instance_Id])     # Start the instance
stop_instances([Instance_Id])      # Stop the instance
terminate_instances([Instance_Id]) # Terminate the instance

print("\nAll tasks completed successfully.")