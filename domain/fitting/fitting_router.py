from typing import List

from fastapi import APIRouter, File, UploadFile
from domain.s3.s3_service import upload_file
from pydantic import BaseModel


class FittingRequestModel(BaseModel):
    model: str
    segmantation: str
    poseSkeleton: str
    keypoints: str
    denseModel: str
    clothesList: List[str]


router = APIRouter(
    prefix="/fast/fitting",
)


@router.post("/")
async def create_virtual_fitting(model: FittingRequestModel):
    # TODO: vton 모델을 통해 사람 이미지에 상의, 하의 또는 한벌옷을 피팅한 이미지 생성
    print(model)

    # s3에 가상 피팅 이미지 업로드 후 반환
    # url = upload_file(image)
    url = "https://fitsta-bucket.s3.ap-northeast-2.amazonaws.com/123.png"
    return {"image": url}


@router.post("/preprocess")
async def regist_fitting_model(image: UploadFile = File(...)):

    origin_img_url = "origin_img_url"
    segmentation = "segmentation22"
    pose_skeleton = "pose_skeleton33"
    keypoints = "keypoint44s"
    dense_model = "dense_model55"

    # TODO: vton 진행시 사용되는 모델이미지 전처리
    return {
        "originImg": origin_img_url,
        "segmentation": segmentation,
        "skeleton": pose_skeleton,
        "keypoints": keypoints,
        "denseModel": dense_model
    }
