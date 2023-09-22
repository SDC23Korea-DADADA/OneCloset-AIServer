from pydantic import BaseModel


class ModelRegistResponse(BaseModel):
    labelMap: str
    skeleton: str
    keypoint: str
    dense: str
    denseNpz: str
