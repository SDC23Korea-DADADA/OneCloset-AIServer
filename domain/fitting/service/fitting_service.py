import uuid
import subprocess
import glob
import os
import logging

# s3를 사용하기 위해 import
from domain.s3.s3_service import upload_general_file

logger = logging.getLogger(__name__)


def preprocess(url):
    origin_path = os.getcwd()
    preprocess_path = "/home/cksghks88/preprocess/"
    dataroot = "/home/cksghks88/data"
    fname = str(uuid.uuid4())[:13]
    logger.info("[Preprocess] fname(uuid): " + fname)

    # preprocess 쉘 스크립트 실행
    os.chdir(preprocess_path)
    exit_code = subprocess.run(
        ["bash", preprocess_path + "preprocess.sh", "--dataroot", dataroot, "--dir", fname, "--url", url])
    os.chdir(origin_path)

    if exit_code == 1:
        logger.error("[Preprocess] preprocess failed")
        raise Exception("preprocess failed")

    file_map = {
        "2.json": "keypoint",
        "4.png": "labelMap",
        "5.jpg": "skeleton",
        "5.png": "dense",
        "uv.npz": "denseNpz"
    }

    url_dict = {}
    for file in glob.glob('*', root_dir=dataroot + "/" + fname):
        postfix = file.split("_")[-1]
        aws_url = upload_general_file(file)

        url_dict[file_map[postfix]] = aws_url

    return url_dict
