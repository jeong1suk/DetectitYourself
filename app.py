import streamlit as st
from torchvision import transforms, models
from PIL import Image
import requests
import tempfile
import numpy as np

# Grad-CAM 라이브러리 임포트
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

###############################################################################
# CONSTANTS
###############################################################################

APP_LOGO = 'img/logo.png'
FAVICON = 'img/favicon.png'
# Hugging Face API 설정
API_URL = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"
API_KEY = st.secrets["huggingface"]["api_key"]

# API 호출 함수
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, data=data)
    return response.json()

# 임시 파일 저장 함수
def fetch_tmp_path(file_data):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(file_data.read())
        return temp.name

###############################################################################
# Streamlit UI 설정
###############################################################################
# Site metadata
icon = Image.open(APP_LOGO)
st.set_page_config(
    page_title="Hugging Face API + GradCAM 시각화",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)
# Logo and header
left_co, right_co = st.columns(2)
with left_co:
    st.image(APP_LOGO)
with right_co:
    """
    # DIY
    ## GenAI Image Detector
    모델: [umm-maybe/AI-image-detector](https://huggingface.co/umm-maybe/AI-image-detector)
    """
# 페이지 헤더
st.title("🚀 Hugging Face API + Grad-CAM")
st.write()
# Subheader
st.write(
    """
    👋 Welcome to DIY! This app detects Real vs GenAI/Fake images using a Convolutional Neural Network.
    
    Powered by [Artifact](https://github.com/awsaf49/artifact), a large-scale dataset with artificial and factual images for synthetic image detection.    
    """
)
st.divider()

# Body 
menu_option = st.radio(
    "What you want?",
    options=["Image", "Fake News(Comming Soon)"],
    horizontal = True
)
###############################################################################
# 모델 준비 (Grad-CAM)
###############################################################################
model = models.resnet50(pretrained=True)
model.eval()

# Grad-CAM 설정
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# 이미지 전처리 파이프라인
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

###############################################################################
# 파일 업로드 및 예측
###############################################################################
if menu_option == "Image":
    uploaded_files = st.file_uploader(
        label="이미지를 업로드하세요 (최대 5개 가능):",
        type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"📸 {len(uploaded_files)} 개의 파일이 업로드되었습니다!")

        # 이미지 경로 및 PIL 객체 생성
        image_paths = [fetch_tmp_path(f) for f in uploaded_files]
        images = [Image.open(p).convert("RGB") for p in image_paths]

        # 예측 결과 시각화
        img_cols = st.columns(len(images))
        for i, img in enumerate(images):
            with img_cols[i]:
                st.image(img, caption=f"업로드된 이미지: {uploaded_files[i].name}", use_container_width=True)

                # Hugging Face API 추론
                result = query(image_paths[i])

                if "error" in result:
                    st.error(f"오류 발생: {result['error']}")
                    continue
                else:
                    label = result[0]["label"]
                    confidence = result[0]["score"]
                    st.write(f"**결과:** {label} ({confidence:.2%})")

                # Grad-CAM 적용
                input_tensor = preprocess(img).unsqueeze(0)
                targets = [ClassifierOutputTarget(281)]  # 원하는 클래스 ID 설정 가능
                try:
                    # Grad-CAM 시각화 생성
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

                    # 원본 이미지 넘파이 변환
                    rgb_img = np.array(img) / 255.0

                    # Grad-CAM 결과를 Pillow 이미지로 변환
                    grayscale_cam_img = Image.fromarray((grayscale_cam * 255).astype(np.uint8))

                    # Grad-CAM 결과 이미지 크기 조정 (Pillow로)
                    resized_cam = grayscale_cam_img.resize((rgb_img.shape[1], rgb_img.shape[0]), Image.ANTIALIAS)

                    # 리사이즈된 Grad-CAM 결과를 NumPy 배열로 변환
                    resized_cam_np = np.array(resized_cam) / 255.0

                    # 시각화 결과 생성
                    visualization = resized_cam_np[..., None] * rgb_img

                    # 결과 표시
                    st.image(visualization, caption="Grad-CAM 시각화 결과", use_container_width=True)
                except Exception as e:
                    st.error(f"시각화 오류: {str(e)}")

elif menu_option == "Fake News(Comming Soon)":
    st.header("📰 Fake News Detection (Coming Soon)")
    st.info("Fake News 탐지 기능은 **추후 개발 예정**입니다. 업데이트를 기대해 주세요!")
