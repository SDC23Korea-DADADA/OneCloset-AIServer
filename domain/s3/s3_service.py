from domain.secret import s3, AWS_S3_BUCKET_NAME, AWS_S3_URL
import uuid
import base64


def upload_file(image):
    # 파일이름앞에 랜덤 uuid 붙이기
    identifier = str(uuid.uuid4())
    filename = identifier + image.filename
    s3.upload_fileobj(image.file, AWS_S3_BUCKET_NAME, filename,
                      ExtraArgs={'ContentType': image.content_type} # 이거 없으면 url로 접근 시 파일 다운로드
                      )
    return AWS_S3_URL + filename

def upload_file_Image(image, filename):
    # S3에 업로드
    identifier = str(base64.urlsafe_b64encode(uuid.uuid4().bytes))
    upload_filename = identifier + filename
    s3.put_object(Bucket=AWS_S3_BUCKET_NAME, Key=upload_filename, Body=image.getvalue(),
                         ContentType='image/png')

    return AWS_S3_URL + upload_filename
