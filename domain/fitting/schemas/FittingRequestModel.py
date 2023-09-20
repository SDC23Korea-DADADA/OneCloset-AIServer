from typing import List
from pydantic import BaseModel


class ClothesInfo(BaseModel):
    type: str
    url: str


class FittingRequestModel(BaseModel):
    model: str
    labelMap: str
    skeleton: str
    keypoint: str
    dense: str
    denseNpz: str
    clothesList: List[ClothesInfo]
