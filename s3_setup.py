import boto3
import os


def download_dir(bucket_name, dir_name):
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=dir_name):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key)  # save to same path


download_dir("daily-bruin", "data")
download_dir("daily-bruin", "images")
