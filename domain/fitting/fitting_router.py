from typing import List, Optional

from fastapi import APIRouter, File, UploadFile

from domain.fitting.schemas.ModelRegistResponse import ModelRegistResponse
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
    clothes_list = request.clothesList
    result_url = None
    for clothes in clothes_list:
        result_url = fitting(request, clothes.type, clothes.url)
        request.model = result_url

    return {"image": result_url}


@router.post("/preprocess", response_model=ModelRegistResponse)
async def regist_fitting_model(request: ImageModel):
    return preprocess(request.image)
