import boto3

AWS_ACCESS_KEY = "AKIAU32EUWK4JSOVIOLZ"
AWS_SECRET_ACCESS_KEY = "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"
AWS_S3_BUCKET_REGION = "ap-northeast-2"
AWS_S3_BUCKET_NAME = "fitsta-bucket"
AWS_S3_URL = "https://fitsta-bucket.s3.ap-northeast-2.amazonaws.com/"

s3 = boto3.client(
    service_name='s3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_S3_BUCKET_REGION
)