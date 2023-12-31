import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.cluster import KMeans
import cv2

import time
import os
from rembg import remove
import colorspacious as cs

# 디바이스 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_directory = os.path.dirname(os.path.realpath(__file__))

materials = {
    0: "코듀로이",
    1: "면",
    2: "니트",
    3: "데님",
    4: "시폰",
    5: "패딩",
    6: "트위드",
    7: "플리스",
    8: "가죽",
}


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class MaterialModel(nn.Module):
    def __init__(self, *args, **kwargs):
        # 1. 모델 구조 정의
        super().__init__(*args, **kwargs)
        self.model = models.efficientnet_v2_m()

        # 2. 모델 가중치 로드
        self.load_weights(os.path.join(current_directory, "models/material_model_state_dict.pth"))
        self.model.eval()

    def load_weights(self, model_path):
        # CPU에서 실행
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    async def preprocess_image(self, image_stream):
        # image_data = await image_stream.read()
        image = Image.open(image_stream).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(480),
            transforms.CenterCrop(480),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0)
        return image


    async def predict(self, image):
        k = 3
        with torch.no_grad():
            outputs = self.model(image)

            # softmax 함수를 사용하여 출력을 확률 분포로 변환
            probs = F.softmax(outputs, dim=1)

            # 확률 값 중에서 top-k 값을 가져옴
            top3_probs, top3_labels = torch.topk(probs, k)

            try:
                top3_labels = [idx.item() for idx in top3_labels[0]]
                top3_probs = [val.item() * 100 for val in top3_probs[0]]  # 확률을 백분율로 변환
                # 결과 출력
                print('재질 추론 결과: ', end=' ')
                for n in range(k):
                    print('%s - %.2f%%' % (materials[top3_labels[n]], top3_probs[n]), end=' ')  # % 기호 추가
                print()
            except:
                # 결과 출력
                print('재질 추론 결과: ', end=' ')
                print('%s - %.2f%%' % (materials[top3_labels[0].item()], top3_probs[0].item()))  # % 기호 추가
            finally:
                return top3_labels[0], top3_probs[0]


class TypeModel(nn.Module):
    def __init__(self, *args, **kwargs):
        # 1. 모델 구조 정의
        super().__init__(*args, **kwargs)
        self.type_model = models.efficientnet_v2_s(weights=None)
        self.type_model.classifier[1].out_features = 18

        # 2. 모델 가중치 로드
        self.load_weights(os.path.join(current_directory, "models/type_model_state_dict.pth"))
        self.type_model.to(device)
        self.type_model.eval()

    def load_weights(self, model_path):
        self.type_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    async def preprocess_image(self, image_stream):
        transform = transforms.Compose([
            transforms.Lambda(lambda img: resize_and_pad(img)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image = Image.open(image_stream)
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            bg = Image.new('RGB', image.size, (255, 255, 255))  # 흰색 배경
            bg.paste(image, mask=a)
            image = bg
        image = transform(image).unsqueeze(0)

        return image

    async def predict(self, image_stream):
        type_int_to_labels = {0: '긴팔티', 1: '반팔티', 2: '셔츠/블라우스', 3: '니트웨어', 4: '후드티', 5: '민소매',
                              6: '긴바지', 7: '반바지', 8: '롱스커트', 9: '미니스커트', 10: '코트',
                              11: '재킷', 12: '점퍼/짚업', 13: '패딩', 14: '가디건', 15: '베스트', 16: '원피스', 17: '점프수트'}

        with torch.no_grad():
            image = await self.preprocess_image(image_stream)
            image = image.to(device)
            outputs = self.type_model(image)
            values, indices = torch.topk(outputs, 3)  # 상위 3개의 확률과 인덱스를 가져옴

            top3_labels = [type_int_to_labels[idx.item()] for idx in indices[0]]
            top3_probs = [val.item() * 100 for val in values[0]]

            # 결과 출력
            print('종류 추론 결과: ', end=' ')
            for n in range(3):
                print('%s - %.2f' % (top3_labels[n], top3_probs[n]), end=' ')
            print()

            return top3_labels[0]


type_model = TypeModel()


async def get_clothes_type(image_stream):
    # TODO: 딥러닝 모델을 통해 의류 종류 추론
    infer_start_time = time.time()
    type = await type_model.predict(image_stream)
    end_time = time.time()
    print("종류 추론 시간: ", end_time - infer_start_time, "seconds")

    return type


async def get_clothes_color(image_stream):
    # TODO: 딥러닝 모델을 통해 의류 색상 추론
    start_time = time.time()
    # 하나의 사진에 대한 추론
    dominant_colors, proportions = extract_dominant_color(image_stream)
    # # 색상 rgb 값 확인용
    for dominant_color, proportion in zip(dominant_colors, proportions):
        print(f"Dominant Color: {dominant_color} {proportion * 100: .2f}%")

    sorted_colors = aggregate_colors(dominant_colors, proportions)

    # 백분율로 변환하고, 문자열로 형식화
    formatted_colors = [f"'{color}': {percentage * 100:.2f}%" for color, percentage in sorted_colors]
    # 결과 문자열 생성
    result_str = ', '.join(formatted_colors)
    print('색상 추론 결과', result_str)

    if(len(sorted_colors) >= 2 and (sorted_colors[0][1] < 0.5)):
        if((sorted_colors[0][0] in ('네이비', '블루', '스카이블루') and sorted_colors[1][0] in ('네이비', '블루', '스카이블루'))
            or (sorted_colors[0][0] in ('핑크', '퍼플') and sorted_colors[1][0] in ('핑크', '퍼플'))
            or (sorted_colors[0][0] in ('그린', '민트', '카키') and sorted_colors[1][0] in ('그린', '민트' '카키'))
            or (sorted_colors[0][0] in ('베이지', '브라운') and sorted_colors[1][0] in ('베이지', '브라운'))
            or (sorted_colors[0][0] in ('옐로우', '오렌지') and sorted_colors[1][0] in ('옐로우', '오렌지'))
            or (sorted_colors[0][0] in ('레드', '와인', '핑크') and sorted_colors[1][0] in ('레드', '와인', '핑크'))
            or (sorted_colors[0][0] in ('화이트', '그레이') and sorted_colors[1][0] in ('화이트', '그레이'))):
            color = sorted_colors[0][0]
        else:
            color = '다채색'
    else:
        color = sorted_colors[0][0]

    print('색상 추론 시간 ', time.time() - start_time, 'seconds')
    return color


async def get_clothes_material(image_stream):
    # TODO: 딥러닝 모델을 통해 의류 재질 추론
    # 시작 시간
    start_time = time.time()
    material_model = MaterialModel()
    # 추론 시작
    infer_start_time = time.time()
    image = await material_model.preprocess_image(image_stream)

    predicted_class, predicted_percent = await material_model.predict(image)

    # 끝 시간
    end_time = time.time()
    # print(predicted_class, " ", predicted_percent)
    print("재질 실행 시간", end_time - start_time, "seconds", ", 추론 시간", end_time - infer_start_time)
    if predicted_percent > 60 : return materials[predicted_class]
    else: return "기타"


def get_clothes_image(image_stream):
    # TODO: 세그멘테이션 모델을 통해 의류 이미지 배경 제거
    input = Image.open(image_stream)

    # 이미지 사이즈가 큰 경우 리사이즈
    if(input.size[0] > 500):
        width = 500
        width_ratio = 500 / float(input.size[0])
        height = int((float(input.size[1])) * width_ratio)
        input = input.resize((width, height))

    output = remove(input)
    # print(output)
    return output


# 가로, 세로 비율 맞춰서 이미지 리사이즈 해주는 함수
def resize_and_pad(image, size=256):
    # 이미지의 원래 크기와 비율 계산
    w, h = image.size
    ratio = w / h

    # 장축을 256에 맞추고, 단축은 비율에 맞춰서 줄임
    if w > h:
        new_w = size
        new_h = int(size / ratio)
    else:
        new_h = size
        new_w = int(size * ratio)

    # 이미지 리사이즈
    resize_transform = transforms.Resize((new_h, new_w))
    image = resize_transform(image)

    # 빈 부분을 검정색으로 채우기
    pad_w = size - new_w
    pad_h = size - new_h

    pad_transform = transforms.Pad((pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2), fill=255)
    image = pad_transform(image)

    return image

# 색상 클래스 정의
color_classes = {
    '블랙': [0, 0, 0], '나이트': [12, 9, 10], '건메탈': [44, 53, 57], '미드나잇': [43, 27, 23], '오일': [59, 49, 49],
    '그레이': [128, 128, 128], '실버': [192, 192, 192], '딤그레이': [105, 105, 105],
    '그린': [0, 128, 0], '다크그린': [0, 100, 0], '옐로우그린': [154, 205, 50],
    '네이비': [0, 0, 128], '다크블루': [0, 0, 139], '미드나잇블루': [25, 25, 112],
    '라벤더': [191, 148, 228],
    '레드': [255, 0, 0],
    '민트': [92, 255, 209],
    '베이지': [212, 184, 134],
    '브라운': [150, 75, 0], '새들브라운': [139, 69, 19],
    '블루': [0, 0, 255], '스틸블루': [70, 130, 180], '터프트블루': [62, 142, 222],
    '스카이블루': [135, 206, 235], '딥스카이블루': [0, 191, 255], '아쿠아': [0, 255, 255],
    '옐로우': [255, 255, 0], '골드': [255, 215, 0],
    # '레몬쉬폰': [255, 250, 205],
    '오렌지': [255, 165, 0], '다크오렌지': [255, 140, 0], '코랄': [255, 127, 80],
    '와인': [114, 47, 55],
    '카키': [138, 147, 82],
    '퍼플': [128, 0, 128],
    # '미디엄퍼플': [147, 112, 219],
    '핑크': [255, 192, 203], '핫핑크': [255, 105, 180], '딥핑크': [255, 20, 147],
    '화이트': [255, 255, 255], '스노우': [255, 250, 250], '화이트스모크': [245, 245, 245] ,'아이보리': [255, 255, 240]
}
class_mapping = {
    '블랙': '블랙', '나이트': '블랙', '건메탈': '블랙', '미드나잇': '블랙', '오일': '블랙',
    '그레이': '그레이', '실버': '그레이', '딤그레이': '그레이', '그린': '그린', '다크그린': '그린', '옐로우그린': '그린',
    '네이비': '네이비', '다크블루': '네이비', '미드나잇블루': '네이비', '라벤더': '라벤더', '레드': '레드', '인디안레드': '레드',
    '민트': '민트', '베이지': '베이지', '아이보리': '베이지', '브라운': '브라운', '새들브라운': '브라운', '블루': '블루', '스틸블루': '블루', '터프트블루': '블루',
    '스카이블루': '스카이블루', '딥스카이블루': '스카이블루', '아쿠아': '스카이블루', '옐로우': '옐로우', '골드': '옐로우', '레몬쉬폰': '옐로우',
    '오렌지': '오렌지', '다크오렌지': '오렌지', '코랄': '오렌지', '와인': '와인', '카키': '카키', '퍼플': '퍼플', '미디엄퍼플': '퍼플',
    '핑크': '핑크', '핫핑크': '핑크', '딥핑크': '핑크', '화이트': '화이트', '스노우': '화이트', '화이트스모크': '화이트'
}


# k-means로 주요 색상 뽑아내기
def extract_dominant_color(image_stream, k=5):
    pil_image = Image.open(image_stream)
    # 이미지의 크기를 256x256로 조정합니다.
    resized_image = pil_image.resize((192, 192))
    # PIL 이미지를 NumPy 배열로 변환합니다.
    image = np.array(resized_image)

    image = enhance_brightness_and_saturation(image)
    # Image.fromarray(image).show()  # 이미지 확인

    # 알파 채널을 사용하여 배경이 아닌 픽셀만 선택하고 알파 채널 제거
    mask = image[:, :, 3] > 0
    image_rgb = image[mask, :3]

    kmeans = KMeans(n_clusters=k, n_init=10, init='k-means++')
    kmeans.fit(image_rgb)
    dominant_colors = kmeans.cluster_centers_

    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    proportions = counts / len(image_rgb)

    # proportions 기준으로 내림차순 정렬
    sorted_indices = np.argsort(proportions)[::-1]
    dominant_colors = dominant_colors[sorted_indices]
    proportions = proportions[sorted_indices]

    # 클러스터링 결과에서 각 클러스터의 중심을 반환
    return dominant_colors, proportions

#  색상 보정 - 명도, 채도 조절
def enhance_brightness_and_saturation(img_array, brightness_factor=1.2, saturation_factor=1.4):
    # RGB에서 HSV로 변환
    img_hsv = cv2.cvtColor(img_array[..., :3], cv2.COLOR_RGB2HSV).astype(float)  # 알파 채널 제외하고 변환
    # 명도, 채도 조절
    img_hsv[:,:,2] = np.clip(img_hsv[:,:,2] * brightness_factor, 0, 255)
    img_hsv[:,:,1] = np.clip(img_hsv[:,:,1] * saturation_factor, 0, 255)

    # HSV에서 RGB로 변환
    img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 원본의 알파 채널이 있으면, 변환된 RGB 이미지에 알파 채널을 다시 추가
    if img_array.shape[2] == 4:
        return np.dstack((img_rgb, img_array[..., 3]))
    else:
        return img_rgb

def closest_color_class(dominant_color):
    # 각 색상 클래스를 Lab 공간으로 변환
    color_keys = list(color_classes.keys())
    lab_colors = np.array([cs.cspace_convert(color_classes[color], "sRGB255", "CIELab") for color in color_keys])

    # 주요 색상을 Lab 공간으로 변환
    dominant_lab = cs.cspace_convert(dominant_color, "sRGB255", "CIELab")

    # 배열 연산으로 각 색상 클래스와 주요 색상 간의 거리 계산
    distances = np.array(
        [cs.deltaE(dominant_lab, lab, input_space="CIELab", uniform_space="JCh") for lab in lab_colors])

    # 가장 가까운 색상 클래스 찾기
    closest_color_key = color_keys[np.argmin(distances)]

    return class_mapping[closest_color_key]


def aggregate_colors(dominant_colors, proportions):
    color_class_mapping = {}

    for color, proportion in zip(dominant_colors, proportions):
        color_class = closest_color_class(color)

        if color_class in color_class_mapping:
            color_class_mapping[color_class] += proportion
        else:
            color_class_mapping[color_class] = proportion

    # Sort by proportions in descending order
    sorted_colors = sorted(color_class_mapping.items(), key=lambda x: x[1], reverse=True)

    return sorted_colors


@singleton
class ClothesClassifier:
    def __init__(self):
        self.model_path = os.path.abspath("domain/clothes/models/cloth_detect_model.pt")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)

    def predict(self, image):
        return self.model(image)


async def is_clothes(image_stream):
    # 시작 시간
    start_time = time.time()
    image = Image.open(image_stream)
    model = ClothesClassifier()

    # 추론 시작
    infer_start_time = time.time()
    results = model.predict(image)  # inference using the loaded model
    confidence_list = results.pandas().xyxy[0]['confidence'].tolist()

    # 끝 시간
    end_time = time.time()
    result = {"isClothes": True, "confidences": confidence_list}
    print("의류 여부 판별 시간: ", (end_time - start_time).__format__(".2f"), "seconds,", ", 추론 시간: ",
          (end_time - infer_start_time).__format__(".2f"), end=" ")

    # 임계값 이상인 것이 있을 경우에 True 반환
    for confidence in confidence_list:
        if confidence >= 0.7:  # 0.7 까지는 옷 데이터가 추출됨
            return result

    result['isClothes'] = False
    return result
