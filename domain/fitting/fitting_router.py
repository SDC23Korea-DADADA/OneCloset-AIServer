from typing import List

from fastapi import APIRouter, File, UploadFile

from domain.fitting.schemas.ModelRegistResponse import ModelRegistResponse
from domain.s3.s3_service import upload_file
from pydantic import BaseModel
from domain.fitting.service.fitting_service import preprocess

from domain.fitting.schemas.FittingRequestModel import FittingRequestModel


# class ClothesInfo(BaseModel):
#     type: str
#     url: str
#
#
# class FittingRequestModel(BaseModel):
#     model: str
#     labelMap: str
#     skeleton: str
#     keypoint: str
#     dense: str
#     denseNpz: str
#     clothesList: List[ClothesInfo]


class ImageModel(BaseModel):
    image: str


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


@router.post("/preprocess", response_model=ModelRegistResponse)
async def regist_fitting_model(request: ImageModel):
    return preprocess(request.image)

    # label_map = "label_map"
    # skeleton = "skeleton"
    # keypoint = "keypoint"
    # dense = "dense"
    # dense_npz = "dense_npz"
    #
    # # TODO: vton 진행시 사용되는 모델이미지 전처리
    # return {
    #     "labelMap": label_map,
    #     "skeleton": skeleton,
    #     "keypoint": keypoint,
    #     "dense": dense,
    #     "denseNpz": dense_npz
    # }
