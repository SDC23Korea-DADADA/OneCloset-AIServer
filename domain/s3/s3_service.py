from domain.secret import s3, AWS_S3_BUCKET_NAME, AWS_S3_URL
import uuid
import base64
import os


def upload_file(image):
    # 파일이름앞에 랜덤 uuid 붙이기
    identifier = str(uuid.uuid4())
    filename = identifier + image.filename
    s3.upload_fileobj(image.file, AWS_S3_BUCKET_NAME, filename,
                      ExtraArgs={'ContentType': image.content_type}  # 이거 없으면 url로 접근 시 파일 다운로드
                      )
    return AWS_S3_URL + filename


def upload_image(image):
    s3.upload_fileobj(image.file, AWS_S3_BUCKET_NAME, image.filename,
                      ExtraArgs={'ContentType': image.content_type}  # 이거 없으면 url로 접근 시 파일 다운로드
                      )
    return AWS_S3_URL + image.filename


def upload_general_file(file, path):
    try:
        file_name = os.path.basename(file)
        with open(path + file, 'rb') as file:
            s3.upload_fileobj(path + file, AWS_S3_BUCKET_NAME, file_name)
        return AWS_S3_URL + file_name
    except FileNotFoundError:
        print(f"The file {file} was not found")
        return None


def upload_file_Image(image, filename):
    # S3에 업로드
    identifier = str(base64.urlsafe_b64encode(uuid.uuid4().bytes))
    upload_filename = identifier + filename
    s3.put_object(Bucket=AWS_S3_BUCKET_NAME, Key=upload_filename, Body=image.getvalue(),
                  ContentType='image/png')

    return AWS_S3_URL + upload_filename
