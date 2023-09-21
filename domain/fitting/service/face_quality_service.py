from PIL import Image
import os
import uuid
import requests
import cv2
import numpy as np

from domain.fitting.service.fitting_service import logger
from domain.s3.s3_service import upload_general_file


def increase_face_quality(request, origin_url):
    result_url = request.model
    mask_url = request.labelMap

    quality_path = "/home/cksghks88/vton/quality/"
    create_directory(quality_path)

    origin_name = str(uuid.uuid4())[:13].replace("-", "") + "_0.jpg"
    result_name = str(uuid.uuid4())[:13].replace("-", "") + "_1.jpg"
    mask_name = str(uuid.uuid4())[:13].replace("-", "") + "_2.jpg"

    download_file(origin_url, quality_path + origin_name)
    download_file(result_url, quality_path + result_name)
    download_file(mask_url, quality_path + mask_name)
    resize_image_file(quality_path + origin_name)
    resize_image_file(quality_path + result_name)
    resize_image_file(quality_path + mask_name)

    # 이미지 불러오기
    origin = cv2.imread(quality_path + origin_name)
    result = cv2.imread(quality_path + result_name)
    mask = cv2.imread(quality_path + mask_name)

    # 출력 이미지를 저장할 배열 생성
    output_img = np.zeros_like(origin)

    # 두 가지 색상 설정 (BGR)
    color1 = [0, 129, 192]
    color2 = [0, 129, 0]

    # 임계값 설정
    threshold = 40

    # 마스크 이미지 순회
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # 마스크 이미지의 픽셀 색상 가져오기
            pixel_color = mask[i, j]

            # 특정 색상과의 거리 계산
            distance1 = np.linalg.norm(pixel_color - color1)
            distance2 = np.linalg.norm(pixel_color - color2)

            # 거리가 임계값 이내인지 확인
            if distance1 < threshold or distance2 < threshold:
                output_img[i, j] = origin[i, j]
            else:
                output_img[i, j] = result[i, j]

    # 결과 이미지 저장
    output_path = str(uuid.uuid4())[:13] + ".jpg"
    cv2.imwrite(quality_path + output_path, output_img)

    # S3에 업로드
    return upload_general_file(output_path, quality_path)



def resize_with_pad(im, target_width, target_height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    target_ratio = target_height / target_width
    im_ratio = im.height / im.width
    if target_ratio > im_ratio:
        # It must be fixed by width
        resize_width = target_width
        resize_height = round(resize_width * im_ratio)
    else:
        # Fixed by height
        resize_height = target_height
        resize_width = round(resize_height / im_ratio)

    image_resize = im.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
    offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB')


def resize_image_file(path):
    image = Image.open(path)
    new = resize_with_pad(image, 384, 512)
    new.save(path)


def create_directory(path):
    """
    주어진 경로에 디렉토리를 생성합니다.
    디렉토리가 이미 존재하면 생성하지 않습니다.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")


def download_file(url, save_path):
    with requests.get(url) as r:
        if r.status_code == 200:
            logger.info("[Download] " + url + " is completed")
            with open(save_path, 'wb') as f:
                f.write(r.content)
        else:
            logger.error("[Download] " + url + " is failed")
