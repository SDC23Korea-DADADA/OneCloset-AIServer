import io

from fastapi import APIRouter, File, UploadFile
import domain.clothes.clothes_service as service
from domain.s3.s3_service import upload_file, upload_file_Image

router = APIRouter(
    prefix="/fast/clothes",
)


@router.post("/check")
async def check_clothes(image: UploadFile = File(...)):

    # TODO: 의류 이미지 판별 로직
    image_data = await image.read()
    image_stream = io.BytesIO(image_data)

    result = await service.is_clothes(image_stream)
    return result


@router.post("/rembg")
async def get_rembg_image(image: UploadFile = File(...)):
    image_data = await image.read()
    image_stream = io.BytesIO(image_data)

    nobg_image, url = await remove_background(image_stream, image.filename)
    return {"image": url}


@router.post("/rembg/info")
async def get_clothes_info(image: UploadFile = File(...)):
    image_data = await image.read()
    image_stream = io.BytesIO(image_data)

    # 배경 제거
    nobg_image, url = await remove_background(image_stream, image.filename)

    # 의류 정보 추출
    nobg_image.seek(0)
    type = await service.get_clothes_type(nobg_image)
    nobg_image.seek(0)
    color = await service.get_clothes_color(nobg_image)
    nobg_image.seek(0)
    material = await service.get_clothes_material_efficient(nobg_image)

    response = {
        "image": url,
        "type": type,
        "color": color,
        "material": material
    }
    return response


@router.post("/test")
async def check_clothes(image: UploadFile = File(...)):
    return upload_file(image)


async def remove_background(image_stream, filename):
    # 의류 이미지 배경 제거
    nobg_image = service.get_clothes_image(image_stream)

    # s3에 배경 제거 이미지 업로드 후 반환
    nobg_image_byte = io.BytesIO()
    nobg_image.save(nobg_image_byte, "PNG")
    url = upload_file_Image(nobg_image_byte, filename)

    return nobg_image_byte, url