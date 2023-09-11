from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from tempfile import NamedTemporaryFile
from typing import IO

router = APIRouter(
    prefix="/fast/clothes",
)

@router.post("/check")
async def check_clothes(image: UploadFile = File(...)):
    # 의류 이미지 판별 로직
    is_clothes = True
    return {"isClothes": is_clothes}

@router.post("/rembg")
async def remove_background(image: UploadFile = File(...)):
    # 의류 이미지에서 배경 제거하는 로직
    image_path = await save_file(image.file)
    return FileResponse(image_path)

@router.post("/info")
async def get_clothes_info(no_bg_image: UploadFile = File(...)):
    # 의류 이미지에서 딥러닝 모델을 통해 종류, 재질, 색상을 추출하는 로직
    response = {
        "type": "바지",
        "color": "블랙",
        "material": "울/캐시미어"
    }
    return response

async def save_file(file: IO):
    # s3 업로드라고 생각해 봅시다. delete=True(기본값)이면
    # 현재 함수가 닫히고 파일도 지워집니다.
    with NamedTemporaryFile("wb", delete=False) as tempfile:
        tempfile.write(file.read())
        return tempfile.name
