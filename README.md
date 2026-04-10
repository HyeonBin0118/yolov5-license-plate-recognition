# 🚗 YOLOv5 기반 번호판 인식 시스템 (License Plate Recognition)

YOLOv5와 OCR을 활용하여 이미지 및 영상에서 차량 번호판을 탐지하고 텍스트를 추출하는 End-to-End 프로젝트입니다.

---

## 📌 프로젝트 소개

본 프로젝트는 객체 탐지 모델(YOLOv5)을 활용하여 차량 번호판을 검출하고,  
OCR을 통해 번호판 텍스트를 자동으로 인식하는 시스템을 구현하는 것을 목표로 합니다.

이미지 및 영상 입력을 기반으로 실시간 번호판 인식이 가능하도록 구성하였습니다.

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
- Bounding Box 시각화 및 정확도 

---

## 📂 프로젝트 구조


├── detect.py
├── models/
├── data/
├── weights/
├── utils/
├── README_KOR.md
└── README_ENG.md


---

## 🚀 실행 방법

### 1. 레포지토리 클론

```bash
git clone https://github.com/HyeonBin0118/yolov5-license-plate-recognition.git
cd yolov5-license-plate-recognition
```
2. 라이브러리 설치
```
pip install -r requirements.txt
```
3. 객체 탐지 실행
```
python detect.py --source path/to/video.mp4 --weights best.pt
```
📸 실행 결과
1. 트레이닝 이미지 500장
<img width="2400" height="1200" alt="result" src="https://github.com/user-attachments/assets/d25c9774-da30-4f09-953e-962763f81201" />
1. 트레이닝 이미지 1500장(이미지 증강활용)
<img width="2400" height="1200" alt="Image" src="https://github.com/user-attachments/assets/475c0a65-8d9b-44af-93aa-975c2bc5ba43" />

🔍 성능 및 결과 분석


👉 학습 과정에서 데이터 정제 및 라벨 오류 수정 후 성능이 개선됨을 확인

🔥 주요 성과
번호판 데이터셋을 활용한 YOLOv5 커스텀 모델 학습
Detection + OCR을 결합한 End-to-End 파이프라인 구축
실시간 영상 기반 번호판 인식 시스템 구현
⚙️ 개선 과정
라벨링 오류(class mismatch) 수정
자동 라벨링 + 수동 검수 진행
데이터 품질 개선을 통한 모델 성능 향상
학습 과정에서 발생한 환경 오류 및 인코딩 문제 해결
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
