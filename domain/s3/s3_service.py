from domain.secret import s3, AWS_S3_BUCKET_NAME, AWS_S3_URL
import uuid


def upload_file(image):
    # 파일이름앞에 랜덤 uuid 붙이기
    identifier = str(uuid.uuid4())
    finename = identifier + image.filename
    s3.upload_fileobj(image.file, AWS_S3_BUCKET_NAME, finename,
                      ExtraArgs={'ContentType': image.content_type} # 이거 없으면 url로 접근 시 파일 다운로드
                      )
    return AWS_S3_URL + finename

