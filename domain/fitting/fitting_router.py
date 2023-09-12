from fastapi import APIRouter, File, UploadFile
from domain.s3.s3_service import upload_file

router = APIRouter(
    prefix="/fast/fitting",
)

@router.post("/")
async def create_virtual_fitting(model, upperwear, underwear, onepiece):
    # TODO: vton 모델을 통해 사람 이미지에 상의, 하의 또는 한벌옷을 피팅한 이미지 생성
    print(model, upperwear, underwear, onepiece)

    # s3에 가상 피팅 이미지 업로드 후 반환
    # url = upload_file(image)
    url = "https://fitsta-bucket.s3.ap-northeast-2.amazonaws.com/123.png"
    return {"image " : url}

