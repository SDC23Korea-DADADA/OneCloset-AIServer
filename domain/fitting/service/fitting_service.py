import uuid
import subprocess
import glob
import os
import logging
import time
import requests
import shutil
from PIL import Image
import cv2
import numpy as np
import copy
import json

# s3를 사용하기 위해 import
from domain.s3.s3_service import upload_general_file

logger = logging.getLogger(__name__)
home_path = "/home/cksghks88/"


def fitting(request, type, cloth_url):
    origin_path = os.getcwd()
    vton_path = home_path + "vton/"
    temp_path = vton_path + "temps/"

    fname = request.labelMap.split("/")[-1].split("_")[0]
    dataroot = vton_path + "inputs/" + fname
    logger.info("[Fitting] fname(uuid): " + fname)

    # input 폴더 생성
    create_default_folder(dataroot + "/dresses")
    create_default_folder(dataroot + "/lower_body")
    create_default_folder(dataroot + "/upper_body")
    clear_path(dataroot, vton_path + "results/unpaired/")

    input_dir = ""
    output_dir = ""
    type_path = ""
    if type == "dress":
        input_dir = dataroot + "/dresses/images/"
        output_dir = vton_path + "results/unpaired/dresses/"
        type_path = "dresses"
    elif type == "lower":
        input_dir = dataroot + "/lower_body/images/"
        output_dir = vton_path + "results/unpaired/lower_body/"
        type_path = "lower_body"
    elif type == "upper":
        input_dir = dataroot + "/upper_body/images/"
        output_dir = vton_path + "results/unpaired/upper_body/"
        type_path = "upper_body"

    # 의류 이미지를 지정된 경로에 저장 & 의류 마스크 이미지 생성
    cloth_fname = str(uuid.uuid4())[:13].replace("-", "") + "_1.jpg"
    download_file(cloth_url, temp_path + cloth_fname)

    shutil.copy(temp_path + cloth_fname, input_dir + cloth_fname)
    image = Image.open(input_dir + cloth_fname)
    new = resize_with_pad(image, 384, 512)
    new.save(input_dir + cloth_fname)
    write_edge(input_dir + cloth_fname, input_dir + os.path.splitext(cloth_fname)[0] + ".png")

    # Model 이미지를 지정된 경로에 저장
    model_fname = fname + "_0.jpg"
    download_file(request.model, temp_path + model_fname)

    shutil.copy(temp_path + model_fname, input_dir + model_fname)
    image = Image.open(input_dir + model_fname)
    new = resize_with_pad(image, 384, 512)
    new.save(input_dir + model_fname)

    # Model 전처리 이미지를 지정된 경로에 저장
    input_dir = input_dir[:-7]
    download_file(request.dense, input_dir + "dense/" + request.dense.split("/")[-1])
    download_file(request.denseNpz, input_dir + "dense/" + request.denseNpz.split("/")[-1])
    download_file(request.keypoint, input_dir + "keypoints/" + request.keypoint.split("/")[-1])
    download_file(request.labelMap, input_dir + "label_maps/" + request.labelMap.split("/")[-1])
    download_file(request.skeleton, input_dir + "skeletons/" + request.skeleton.split("/")[-1])

    # 추론에 필요한 텍스트 파일 생성
    typeNum = type == "upper" and 0 or type == "lower" and 1 or 2
    with open(dataroot + "/test_pairs_paired.txt", "w") as f:
        f.write(fname + "_0.jpg\t" + cloth_fname + "\t" + str(typeNum))
    with open(dataroot + type_path + "/test_pairs_unpaired.txt", "w") as f:
        f.write(fname + "_0.jpg\t" + cloth_fname)

    # preprocess 쉘 스크립트 실행
    os.chdir(vton_path)
    subprocess.run(
        ["bash", vton_path + "test.sh", "--dataroot", dataroot])
    os.chdir(origin_path)

    # 가상피팅 결과 반환
    output_fname = fname + "_0.jpg"
    aws_url = upload_general_file(output_fname, output_dir)
    return aws_url


def preprocess(url):
    origin_path = os.getcwd()
    preprocess_path = home_path + "preprocess/"
    dataroot = home_path + "data"
    fname = str(uuid.uuid4())[:13].replace("-", "")
    logger.info("[Preprocess] fname(uuid): " + fname)

    # preprocess 쉘 스크립트 실행
    os.chdir(preprocess_path)
    subprocess.run(
        ["bash", preprocess_path + "preprocess.sh", "--dataroot", dataroot, "--dir", fname, "--url", url])
    os.chdir(origin_path)

    file_map = {
        "2.json": "keypoint",
        "4.png": "labelMap",
        "5.jpg": "skeleton",
        "5.png": "dense",
        "uv.npz": "denseNpz"
    }

    url_dict = {}
    for file in glob.glob('*', root_dir=dataroot + "/" + fname + "/outputs"):
        postfix = file.split("_")[-1]
        aws_url = upload_general_file(file, dataroot + "/" + fname + "/outputs/")
        url_dict[file_map[postfix]] = aws_url

    logger.info("[Preprocess] url_dict: " + str(url_dict))
    return url_dict


def download_file(url, save_path):
    with requests.get(url) as r:
        if r.status_code == 200:
            logger.info("[Download] " + url + " is completed")
            with open(save_path, 'wb') as f:
                f.write(r.content)
        else:
            logger.error("[Download] " + url + " is failed")


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


def create_default_folder(path):
    create_directory(path + "/dense")
    create_directory(path + "/images")
    create_directory(path + "/keypoints")
    create_directory(path + "/label_maps")
    create_directory(path + "/masks")
    create_directory(path + "/skeletons")


def clear_path(input_path, output_path):
    files = glob.glob(f'{input_path}/*/*/*.*')
    for f in files:
      os.remove(f)

    files = glob.glob(f'{output_path}/*/*/*.*')
    for f in files:
      os.remove(f)

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


def otsu(img, n, x):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, n, x)
    return thresh


def contour(img):
    edges = cv2.dilate(cv2.Canny(img, 200, 255), None)
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    masked = cv2.drawContours(mask, [cnt], -1, 255, -1)
    return masked


def get_cloth_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    return mask


def write_edge(C_path, E_path):
    img = cv2.imread(C_path)
    res = get_cloth_mask(img)
    if (np.mean(res) < 100):
        ot = otsu(img, 11, 0.6)
        res = contour(ot)
    cv2.imwrite(E_path, res)
