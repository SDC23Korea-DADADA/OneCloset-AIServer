from domain.secret import s3, AWS_S3_BUCKET_NAME, AWS_S3_URL


def upload_file(image):
    # 파일이름앞에 랜덤 uuid 붙이기
    s3.upload_fileobj(image.file, AWS_S3_BUCKET_NAME, image.filename)
    return AWS_S3_URL + image.filename
