from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from domain.clothes import clothes_router
from domain.fitting import fitting_router

app = FastAPI()

origins = [
    "http://127.0.0.1:8080",  # 요청하는 spring 도메인
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(clothes_router.router)
app.include_router(fitting_router.router)