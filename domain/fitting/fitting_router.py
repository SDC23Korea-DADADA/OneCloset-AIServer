from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse

router = APIRouter(
    prefix="/fast/fitting",
)

@router.post("/")
async def create_virtual_fitting(images: list[UploadFile]):
    # 모델 이미지에 상의, 하의를 피팅한 이미지를 반환하는 로직
    return {"filenames " : [image.filename for image in images]}

