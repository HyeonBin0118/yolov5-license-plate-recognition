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

- 차량 번호판 실시간 검출
- 검출된 번호판 영역 Crop
- OCR 기반 텍스트 추출
- 이미지 및 영상 입력 지원
- Bounding Box 시각화 및 정확도 표시



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



📸 실행 결과
1. 트레이닝 이미지 500장 epoc 100
<img width="2400" height="1200" alt="result" src="https://github.com/user-attachments/assets/d25c9774-da30-4f09-953e-962763f81201" />

---

2. 트레이닝2 이미지 1500장(이미지 증강활용) epoc 100
(적용 항목은 수평 뒤집기, 회전. 명암/대비조절, 색상/채도/명도, 이동/확대/축소, 가우시안 블러, 감마값 조절)
<img width="2400" height="1200" alt="Image" src="https://github.com/user-attachments/assets/475c0a65-8d9b-44af-93aa-975c2bc5ba43" />

🔍 성능 및 결과 분석


👉 데이터 증강(Data Augmentation)을 활용하여 학습 데이터의 다양성을 확장하였다. 특히 번호판 인식 특성상 실제 환경에서 발생하는 다양한 변수(조명 변화, 햇빛 반사, 기울기, 원근 왜곡, 부분 가림 등)를 반영하기 위해 밝기 조절, 회전, 스케일 변환, 노이즈 추가 등의 증강 기법을 적용하였다. 그 결과 다음과 같은 결과물을 얻었다.
mAP@0.5: ~0.88 → ~0.90+ 유지
mAP@0.5:0.95: ~0.55 → ~0.48~0.5 유지 
큰 상승폭은 아니나, 이전 모델 대비 precision이 향상되었고, 전체적인 학습 안정성이 개선되었다. 다만 validation object loss가 후반부에 증가하는 경향을 보임. 약한 overfitting이 발생하여, 이후 epoch 조정이나 학습 데이터 보충을 통해 개선할 여지가 있으나 현 보유 학습 데이터만으로는 연속적인 trade-off 관계에 있음.


🔥 주요 성과
번호판 데이터셋을 활용한 YOLOv5 커스텀 모델 학습
Detection + OCR을 결합한 End-to-End 파이프라인 구축
실시간 영상 기반 번호판 인식 시스템 구현


⚙️ 개선 과정
불안정한 그래프 형상 최적화를 위해 이미지 증강을 통한 데이터 확보
자동 라벨링 + 수동 검수 진행 
학습 과정에서 발생한 환경 오류 및 인코딩 문제 해결

---
📌 향후 개선 방향
OCR 인식 정확도 향상
다양한 환경(야간, 흐림 등) 데이터 추가
실시간 처리 속도 최적화
웹 서비스 형태로 배포
🙋‍♂️ 프로젝트를 통해 배운 점
모델 성능은 데이터 품질에 크게 영향을 받는다는 점
단순 학습이 아닌 데이터 전처리 및 개선 과정의 중요성
Detection + OCR 파이프라인 설계 경험
👨‍💻 Author
HyeonBin Kim(HyeonBin0118)
