import time
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile
from domain.fitting.schemas.ModelRegistResponse import ModelRegistResponse
from domain.fitting.service.face_quality_service import increase_face_quality
from domain.s3.s3_service import upload_file
from pydantic import BaseModel
from domain.fitting.service.fitting_service import preprocess, fitting
from domain.fitting.schemas.FittingRequestModel import FittingRequestModel


class ImageModel(BaseModel):
    image: str


router = APIRouter(
    prefix="/fast/fitting",
)


@router.post("/")
async def create_virtual_fitting(request: Optional[FittingRequestModel] = None):
    start_time = time.time()

    model_url = request.model
    clothes_list = request.clothesList
    result_url = None
    for clothes in clothes_list:
        result_url = fitting(request, clothes.type, clothes.url)
        request.model = result_url

    high_quality_url = increase_face_quality(request, model_url)

    end_time = time.time()
    print("가상피팅 처리시간 :", end_time - start_time, "seconds")

    return {"image": high_quality_url}


@router.post("/preprocess", response_model=ModelRegistResponse)
async def regist_fitting_model(request: ImageModel):
    start_time = time.time()

    result = preprocess(request.image)

    end_time = time.time()
    print("Model 전처리 처리시간 :", end_time - start_time, "seconds")
    return result
