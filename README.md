# 🚗 YOLOv5 기반 번호판 인식 시스템 (License Plate Recognition)

YOLOv5와 OCR을 활용하여 이미지 및 영상에서 차량 번호판을 탐지하고 텍스트를 추출하는 End-to-End 프로젝트입니다.

---

## 📌 프로젝트 소개

본 프로젝트는 객체 탐지 모델(YOLOv5)을 활용하여 차량 번호판을 검출하고,  
OCR을 통해 번호판 텍스트를 자동으로 인식하는 시스템을 구현하는 것을 목표로 합니다.
디텍팅과 문자추출(OCR)과정을 한 번에 수행하는 것을 목표로 코드를 구성합니다.

---

## 🛠️ 기술 스택

- Python  
- YOLOv5  
- OpenCV  
- OCR (Tesseract / PaddleOCR)  
- PyTorch  

---

## 🎯 주요 기능

- YOLOv5기반 차량 번호판 실시간 검출
- 검출된 번호판 영역 Crop
- PaddleOCR 기반 텍스트 추출
- 한글 번호판 텍스트 출력 지원
- 이미지 및 영상 입력 지원

---
🧠 시스템 파이프라인
```
Input (Image/Video)
        ↓
YOLOv5 (번호판 검출)
        ↓
Plate Crop
        ↓
Image Preprocessing
        ↓
PaddleOCR (텍스트 인식)
        ↓
Result Visualization
```
---

## 🚀 실행 방법

### 1. 레포지토리 클론

```bash
git clone https://github.com/HyeonBin0118/yolov5-license-plate-recognition.git
cd yolov5-license-plate-recognition
cd custom_code
```
2. 라이브러리 설치
```
pip install -r requirements.txt
```
3. 객체 탐지 실행
```
python detect_plate.py --source ..\test_data\test.mp4 --weights ..\weight\best.pt
```
🧩 핵심 코드 구현
1. 번호판 검출 → OCR 연결 파이프라인
```python
# 번호판 영역 크롭
x1, y1, x2, y2 = map(int, xyxy)
pad = 5
x1 = max(x1 - pad, 0)
y1 = max(y1 - pad, 0)
x2 = min(x2 + pad, im0.shape[1])
y2 = min(y2 + pad, im0.shape[0])

plate_img = im0[y1:y2, x1:x2]

plate_text = ""

# OCR 처리
if plate_img is not None and plate_img.size > 0:
    try:
        processed = preprocess_plate(plate_img)

        result = ocr.ocr(processed, cls=True)

        if result and result[0]:
            for line in result[0]:
                text, ocr_conf = line[1]
                if ocr_conf > 0.4:
                    plate_text += text

    except Exception as e:
        plate_text = ""
```
2. OCR 성능 향상을 위한 전처리
```python
def preprocess_plate(plate_img):
    h, w = plate_img.shape[:2]

    # 작은 번호판 인식률 향상을 위한 업스케일링
    plate_img = cv2.resize(
        plate_img,
        (w * 2, h * 2),
        interpolation=cv2.INTER_CUBIC
    )

    # 노이즈 제거
    plate_img = cv2.GaussianBlur(plate_img, (3, 3), 0)

    return plate_img
```
3. 한글 텍스트 렌더링 (OpenCV + PIL)
```python
# OpenCV → PIL 변환
img_pil = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)

font = ImageFont.truetype(font_path, 45)

# 텍스트 출력
draw.text(
    (x1, max(y1 - 70, 0)),
    plate_text,
    font=font,
    fill=(0, 255, 0)
)

# PIL → OpenCV 복원
im0 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
```
---
📸 실행 결과
1. 트레이닝 이미지 500장 epoc 100
<img width="2400" height="1200" alt="result" src="https://github.com/user-attachments/assets/d25c9774-da30-4f09-953e-962763f81201" />



2. 트레이닝2 이미지 1500장(이미지 증강활용) epoc 100

(적용 항목은 수평 뒤집기, 회전. 명암/대비조절, 색상/채도/명도, 이동/확대/축소, 가우시안 블러, 감마값 조절)
<img width="2400" height="1200" alt="Image" src="https://github.com/user-attachments/assets/475c0a65-8d9b-44af-93aa-975c2bc5ba43" />

🔍 성능 및 결과 분석
---
👉초기 학습 결과 전체적으로 모든 지표가 튀는 형상을 보이고 mAP측면에서 매우 불안정한 지표를 보인다. 이는 모델이 안정적으로 수렴하고 있지 않음을 의미한다. 원인으로는 번호판 데이터의 특성상 촬영 환경에 따라 입력 이미지의 편차가 크게 발생하기 때문으로 판단된다. 실제 번호판은 다양한 각도, 거리, 조명(야간/주간), 반사, 오염, 그리고 모션 블러 등의 영향을 받으며, 이러한 요소들이 데이터셋에 충분히 반영되지 않을 경우 특정 환경에서는 높은 성능을 보이지만 다른 환경에서는 성능이 급격히 저하되는 문제가 발생한다.

이러한 문제를 해결하기 위해 데이터 다양성을 확보하는 방향으로 접근하였으며, 밝기 변화, 기울기 변환, 노이즈 추가 등의 이미지 증강 기법을 적용하여 다양한 실제 상황을 모사한 데이터를 추가로 학습시켰다. 이를 통해 모델이 특정 조건에 과도하게 적응하는 것을 방지하고, 보다 일반화된 특징을 학습하도록 유도하였다.

결과적으로 mAP 지표를 보면 이전 대비 상승하며 동시에 변동성이 감소하여 모델의 전반적인 성능과 안정성이 개선되었음을 확인할 수 있었다. 이는 데이터 증강을 통해 다양한 환경에 대한 일반화 성능이 향상된 결과로 해석된다. 다만, 학습 후반부에서 val/obj_loss가 완만하게 증가하는 경향이 관찰되어, 모델이 점차 학습 데이터에 과적합되기 시작하는 초기 징후 또한 확인되었다. 

추가적인 epoch 조정 및 데이터 확장을 통해 성능 개선의 여지는 존재하나, 번호판 데이터는 개인정보와 밀접하게 연관되어 있어 공개 데이터 확보에 현실적인 제약이 따른다. 본 프로젝트에서는 이러한 제한된 데이터 환경을 전제로 학습을 진행하였으며, 그 결과 모델은 성능 향상과 과적합 사이의 trade-off를 보였다. 따라서 현재 성능은 주어진 데이터 조건 하에서의 합리적인 결과로 해석할 수 있다.



---
🎥 Demo

![Video Project 4](https://github.com/user-attachments/assets/4c73dc86-93e7-4af8-bad9-d6711e7b0743)

---



⚙️ 개선과정

---
초기 학습에서 mAP 및 validation loss가 요동치는 현상이 발생하였고, 이를 데이터 다양성 부족 문제로 판단하였다. 이에 따라 밝기, 기울기, 노이즈 등의 이미지 증강을 적용하여 재학습을 진행하였고, 그 결과 학습 곡선이 보다 안정적으로 수렴하도록 개선하였다.


자동 라벨링을 통해 데이터를 빠르게 구축한 뒤, 수동 검수를 병행하여 라벨의 정확도를 개선하였다. 이를 통해 잘못된 라벨로 인한 학습 성능 저하를 방지하였다.


초기 OCR 단계에서 한글 번호판 인식이 제대로 이루어지지 않는 문제가 발생하였다. 이를 해결하기 위해 PaddleOCR로 엔진을 교체하였으며, 한국어 환경에 적합한 OCR을 적용하여 인식 정확도를 개선하였다.


번호판 영역이 작거나 해상도가 낮은 경우 OCR 인식률이 급격히 저하되는 문제가 발생하였다. 특히 영상 프레임에서 크롭된 번호판은 정보 손실이 발생하기 쉬워 인식이 불안정하였다. 이를 해결하기 위해 이미지 업스케일링 및 노이즈 제거 기반 전처리를 적용하여 OCR 입력 품질을 개선하였다.


ocr로 추출한 텍스트가 영상에 정상적으로 출력되지 않는 문제가 발생하였다. 이를 해결하기 위해 openCV와 PIL을 활용한 랜더링 방식을 적용하여 한글 텍스트를 안정적으로 출력하도록 개선하였다. 


---
📌 향후 개선 방향
---
OCR 인식 정확도 향상

다양한 환경(야간, 흐림 등) 데이터 추가

실시간 처리 속도 최적화

웹 서비스 형태로 배포

---
👨‍💻 Author
---
HyeonBin Kim(HyeonBin0118)
