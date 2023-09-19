import cv2
from fastapi import UploadFile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

import time
import os
import io
from io import BytesIO

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 학습한 모델과 동일한 모델 정의
class ColorModel(nn.Module):
    def __init__(self):
        super(ColorModel, self).__init__()
        # 숫자를 라벨로 변환
        color_labels_to_int = {'블랙': 1, '그레이': 2, '그린': 3, '네이비': 4, '라벤더': 5,
                               '레드': 6, '민트': 7, '베이지': 8, '브라운': 9, '블루': 10,
                               '스카이블루': 11, '옐로우': 12, '오렌지': 13, '와인': 14, '카키': 15,
                               '퍼플': 16, '핑크': 17, '화이트': 18, '다채색': 19}

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(color_labels_to_int))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MaterialModel(nn.Module):
    def __init__(self, *args, **kwargs):
        # 1. 모델 구조 정의
        super().__init__(*args, **kwargs)
        self.model = models.mobilenet_v3_small()
        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)

        # 2. 모델 가중치 로드
        current_directory = os.path.dirname(os.path.realpath(__file__))
        self.load_weights(os.path.join(current_directory, "material_model_state_dict.pth"))
        self.model.eval()

    def load_weights(self, model_path):
        # CPU에서 실행
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    async def preprocess_image(self, image_stream):
        # await image_file.seek(0)
        # image_data = await image_file.read()
        image = Image.open(image_stream).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0)
        return image

    async def predict(self, image):
        material_int_to_labels = {
            0: "퍼",
            1: "면",
            2: "니트",
            3: "데님",
            4: "시폰",
            5: "패딩",
            6: "트위드",
            7: "플리스",
            8: "가죽",
            9: "코듀로이",
        }
        k = 3
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = outputs.max(1)

            values, indices = torch.topk(outputs, k)
            top3_labels = [idx.item() for idx in indices[0]]
            top3_probs = [val.item() * 100 for val in values[0]]

            # 결과 출력
            print('재질 추론 결과: ', end=' ')
            for n in range(k):
                print('%s - %.2f' % (material_int_to_labels[top3_labels[n]], top3_probs[n]), end=' ')
            print()
            material = top3_labels[0]
            return material


async def get_clothes_type(image_stream):
    # TODO: 딥러닝 모델을 통해 의류 종류 추론
    type_int_to_labels = {0: '긴팔티', 1: '반팔티', 2: '셔츠/블라우스', 3: '니트웨어', 4: '후드티', 5: '민소매',
                     6: '긴바지', 7: '반바지', 8: '롱스커트', 9: '미니스커트', 10: '코트',
                     11: '재킷', 12: '점퍼/짚업', 13: '패딩', 14: '가디건', 15: '베스트', 16: '원피스', 17: '점프수트'}

    transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    start_time = time.time()

    current_directory = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_directory, "type_model_state_dict.pth")

    model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))  # 학습된 모델의 매개변수 불러오기
    model.to(device)
    model.eval()

    image = Image.open(image_stream)
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        bg = Image.new('RGB', image.size, (255, 255, 255))  # 흰색 배경
        bg.paste(image, mask=a)
        image = bg
    image = transform(image).unsqueeze(0)

    infer_start_time = time.time()
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        values, indices = torch.topk(outputs, 3)  # 상위 3개의 확률과 인덱스를 가져옴

        top3_labels = [type_int_to_labels[idx.item()] for idx in indices[0]]
        top3_probs = [val.item() * 100 for val in values[0]]

        # 결과 출력
        print('종류 추론 결과: ', end=' ')
        for n in range(3):
            print('%s - %.2f' % (top3_labels[n], top3_probs[n]), end=' ')
        print()
        type = top3_labels[0]

    end_time = time.time()
    print("종류 실행 시간", end_time - start_time, "seconds", ", 추론 시간", end_time - infer_start_time, "seconds")

    # 라벨을 숫자로 변환
    return type


async def get_clothes_color(image_stream):
    # TODO: 딥러닝 모델을 통해 의류 색상 추론
    # 라벨을 숫자로 변환
    color_int_to_labels = {1: '블랙', 2: '그레이', 3: '그린', 4: '네이비', 5: '라벤더',
                           6: '레드', 7: '민트', 8: '베이지', 9: '브라운', 10: '블루',
                           11: '스카이블루', 12: '옐로우', 13: '오렌지', 14: '와인', 15: '카키',
                           16: '퍼플', 17: '핑크', 18: '화이트', 19: '다채색'}

    # 시작 시간
    start_time = time.time()

    current_directory = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_directory, "color_model_state_dict.pth")

    # 모델을 평가 모드로 설정
    model = ColorModel()

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))  # 학습된 모델의 매개변수 불러오기
    model.to(device)
    model.eval()

    # 테스트 데이터 생성
    transform = transforms.Compose(
        [transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = Image.open(image_stream)
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        bg = Image.new('RGB', image.size, (255, 255, 255))  # 흰색 배경
        bg.paste(image, mask=a)
        image = bg
    image = transform(image).unsqueeze(0)

    # 추론 시작
    infer_start_time = time.time()
    # 추론
    with torch.no_grad():
        image = image.to(device)

        outputs = model(image)
        values, indices = torch.topk(outputs, 3)  # 상위 3개의 확률과 인덱스를 가져옴

        top3_labels = [color_int_to_labels[idx.item()] for idx in indices[0]]
        top3_probs = [val.item() * 100 for val in values[0]]

        # 결과 출력
        print('색상 추론 결과: ', end=' ')
        for n in range(3):
            print('%s - %.2f' % (top3_labels[n], top3_probs[n]), end=' ')
        print()
        color = top3_labels[0]

    # 끝 시간
    end_time = time.time()
    print("색상 실행 시간", end_time - start_time, "seconds", ", 추론 시간", end_time - infer_start_time)

    return color


async def get_clothes_material(image_stream):
    # TODO: 딥러닝 모델을 통해 의류 재질 추론
    # 시작 시간
    start_time = time.time()
    resnet = MaterialModel()
    # 추론 시작
    infer_start_time = time.time()
    image = await resnet.preprocess_image(image_stream)

    predicted_class = await resnet.predict(image)
    materials = {
        0: "퍼",
        1: "면",
        2: "니트",
        3: "데님",
        4: "시폰",
        5: "패딩",
        6: "트위드",
        7: "플리스",
        8: "가죽",
        9: "코듀로이",
    }
    # 끝 시간
    end_time = time.time()
    print("재질 실행 시간", end_time - start_time, "seconds", ", 추론 시간", end_time - infer_start_time)
    return materials[predicted_class]


def get_clothes_image(file: UploadFile):
    # TODO: 세그멘테이션 모델을 통해 의류 이미지 배경 제거
    return
