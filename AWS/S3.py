# ===========================================================
# AWS S3 HANDS-ON TUTORIAL SCRIPT
# This file contains multiple S3 automation examples using boto3.
# The script is intended as a living reference that you can
# revise and expand throughout your learning journey.
# ===========================================================

import boto3
import os
import sys
from botocore.exceptions import ClientError

# -----------------------------------------------------------
# Global configuration values used across the script
# -----------------------------------------------------------
BUCKET_NAME = 'arupreza-test-bucket-12345'
REGION = 'ap-southeast-2'

# -----------------------------------------------------------
# Create reusable boto3 interfaces
# -----------------------------------------------------------
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')


# ===========================================================
def create_bucket(bucket_name, region=REGION):
    # Creates a new S3 bucket in the specified AWS region.
    try:
        response = s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={
                'LocationConstraint': region
            }
        )
        print(f"Bucket '{bucket_name}' created successfully")
        return response
    except ClientError as e:
        print("AWS Error:", e.response['Error']['Message'])
    except Exception as e:
        print("Unexpected error:", e)

# -----------------------------------------------------------
# Explanation and Use
#
# - This function creates an S3 bucket.
# - The name must be globally unique across AWS.
# - Region is passed using LocationConstraint.
#
# Typical use:
# create_bucket("my-unique-bucket-name")
#
# You run this once before performing uploads.
# -----------------------------------------------------------


# ===========================================================
def upload_file(file_path, bucket_name=BUCKET_NAME, object_name=None):

    # Uploads a single file to S3 bucket.
    try:
        if object_name is None:
            object_name = os.path.basename(file_path)

        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"Uploaded '{file_path}' as '{object_name}'")

    except ClientError as e:
        print("Upload failed:", e.response['Error']['Message'])
    except Exception as e:
        print("Unexpected error:", e)

# -----------------------------------------------------------
# Explanation and Use
#
# - Sends one local file to S3.
# - If you don't give object_name, the original filename is used.
# - upload_file() is high-level and manages multipart upload.
#
# Example usage:
#
# upload_file("data/report.json")
# upload_file("data/report.json", object_name="backup.json")
#
# This is the most common operation in real projects.
# -----------------------------------------------------------


# ===========================================================
def list_objects(bucket_name=BUCKET_NAME):

    # Lists all object keys stored in S3 bucket.
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)

        print("Objects in bucket:")
        for obj in response.get('Contents', []):
            print(" -", obj['Key'])

    except ClientError as e:
        print("List operation failed:", e.response['Error']['Message'])
    except Exception as e:
        print("Unexpected error:", e)

# -----------------------------------------------------------
# Explanation and Use
#
# - Retrieves metadata for objects inside a bucket.
# - Iterates through the 'Contents' list and prints keys.
#
# Example:
# list_objects()
#
# You use this to verify:
# - uploads succeeded
# - prefixes contain expected files
#
# Very useful while debugging.
# -----------------------------------------------------------


# ===========================================================
def download_file(object_name, file_path, bucket_name=BUCKET_NAME):

    # Downloads a specific S3 object to a local file.
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        s3_client.download_file(bucket_name, object_name, file_path)
        print(f"Downloaded '{object_name}' to '{file_path}'")

    except ClientError as e:
        print("Download failed:", e.response['Error']['Message'])
    except Exception as e:
        print("Unexpected error:", e)

# -----------------------------------------------------------
# Explanation and Use
#
# - Brings a file from S3 to your machine.
# - Automatically creates directories if needed.
#
# Example usage:
#
# download_file("backup.json", "downloads/backup.json")
#
# This allows:
# - restoring backups
# - fetching datasets
# - syncing config files
# -----------------------------------------------------------


# ===========================================================
def upload_dir(dir_path, bucket_name=BUCKET_NAME, s3_prefix=""):

    # Recursively uploads all files from a directory.
    try:
        for root, dirs, files in os.walk(dir_path):
            for file in files:

                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, dir_path)

                s3_key = os.path.join(s3_prefix, rel_path).replace("\\", "/")

                s3_client.upload_file(local_path, bucket_name, s3_key)
                print(f"Uploaded: {local_path} -> {s3_key}")

    except Exception as e:
        print("Directory upload failed:", e)

# -----------------------------------------------------------
# Explanation and Use
#
# - This function uploads MANY files at once.
# - os.walk() iterates through subfolders.
# - The prefix organizes them inside S3.
#
# Example:
#
# upload_dir("./Downloaded", s3_prefix="backups/project_v1")
#
# Practical uses:
# - backup folders
# - upload experiment outputs
# - move entire project results
# -----------------------------------------------------------


# ===========================================================
def download_dir(bucket_name, s3_prefix, local_path):

    # Downloads all files with the prefix using pagination.
    try:
        paginator = s3_client.get_paginator("list_objects_v2")

        for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):

            for item in result.get("Contents", []):

                s3_key = item["Key"]
                rel_path = os.path.relpath(s3_key, s3_prefix)

                local_file = os.path.join(local_path, rel_path)

                os.makedirs(os.path.dirname(local_file), exist_ok=True)

                if not s3_key.endswith('/'):
                    s3_client.download_file(bucket_name, s3_key, local_file)
                    print(f"Downloaded: {s3_key} -> {local_file}")

    except Exception as e:
        print("Directory download failed:", e)

# -----------------------------------------------------------
# Explanation and Use
#
# - S3 can return only limited results per call.
# - Paginator is required for large prefixes.
# - This rebuilds S3 folder structure locally.
#
# Example usage:
#
# download_dir("my-bucket", "backups/project_v1/", "local_backups")
#
# This is useful for:
# - syncing S3 data to new machines
# - migrating projects
# -----------------------------------------------------------


# ===========================================================
def delete_s3_files(bucket_name, s3_prefix):

    # Deletes all objects matching a prefix.
    try:
        paginator = s3_client.get_paginator("list_objects_v2")

        for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):

            for item in result.get("Contents", []):

                s3_key = item["Key"]

                s3_client.delete_object(
                    Bucket=bucket_name,
                    Key=s3_key
                )

                print("Deleted:", s3_key)

    except Exception as e:
        print("Delete operation failed:", e)

# -----------------------------------------------------------
# Explanation and Use
#
# - Bulk removes files in S3.
# - Useful for cleaning backups or old outputs.
#
# Example:
#
# delete_s3_files("my-bucket", "tmp/")
#
# You use this before re-uploading fresh data.
# -----------------------------------------------------------


# ===========================================================
def empty_s3_bucket(bucket_name):

    # Empties entire bucket including versions.
    try:
        print(f"Emptying bucket '{bucket_name}'...")

        bucket = s3_resource.Bucket(bucket_name)
        bucket.object_versions.delete()

        print(f"Bucket '{bucket_name}' is now empty.")

    except Exception as e:
        print("Empty bucket failed:", e)

# -----------------------------------------------------------
# Explanation and Use
#
# - This uses the boto3.resource API.
# - object_versions.delete() handles pagination internally.
#
# Example:
# empty_s3_bucket("my-bucket")
#
# This is mandatory before deleting a bucket.
# -----------------------------------------------------------


# ===========================================================
def delete_entire_s3_bucket(bucket_name):

    # Deletes the bucket structure itself.
    try:
        print(f"Deleting bucket '{bucket_name}'...")

        bucket = s3_resource.Bucket(bucket_name)
        bucket.delete()

        print(f"Bucket '{bucket_name}' deleted successfully")

    except ClientError as e:
        print("AWS Error:", e.response['Error']['Message'])
    except Exception as e:
        print("Unexpected error:", e)

# -----------------------------------------------------------
# Explanation and Use
#
# - Permanently removes bucket from AWS account.
# - Requires it to be empty first.
#
# Example:
# delete_entire_s3_bucket("my-bucket")
#
# Used for:
# - test buckets
# - temporary project storage
# -----------------------------------------------------------