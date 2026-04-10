"""
detect.py
---------
YOLOv5 기반 한국 차량 번호판 검출 및 OCR 인식 스크립트.

주요 기능:
  - YOLOv5 커스텀 모델로 영상/이미지에서 번호판 영역 검출
  - 검출된 번호판을 PaddleOCR로 텍스트 인식
  - 인식 결과를 영상 위에 한글 폰트로 오버레이하여 저장
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch
from paddleocr import PaddleOCR

# PaddleOCR 초기화 (한국어 모드, 텍스트 방향 자동 감지 활성화)
ocr = PaddleOCR(lang='korean', use_angle_cls=True, show_log=False)

import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image

# 한글 출력을 위한 윈도우 시스템 폰트 경로
font_path = "C:/Windows/Fonts/malgun.ttf"

# 현재 파일 기준으로 YOLOv5 루트 경로 설정
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


def preprocess_plate(plate_img):
    """
    번호판 이미지 전처리 함수.

    OCR 인식률을 높이기 위해 이미지를 확대하고 노이즈를 제거한다.
    PaddleOCR은 BGR 3채널 컬러 이미지를 입력으로 요구하므로,
    grayscale 변환 없이 컬러 상태를 유지하여 반환한다.

    Args:
        plate_img: 번호판 영역 크롭 이미지 (BGR 3채널)

    Returns:
        전처리된 BGR 3채널 이미지
    """
    h, w = plate_img.shape[:2]

    # 작은 번호판 이미지의 인식률 향상을 위해 2배 확대
    plate_img = cv2.resize(plate_img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # 가우시안 블러로 노이즈 제거
    plate_img = cv2.GaussianBlur(plate_img, (3, 3), 0)

    return plate_img


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",       # 모델 가중치 파일 경로
    source=ROOT / "data/images",        # 입력 소스 (파일, 디렉토리, URL, 웹캠 등)
    data=ROOT / "data/coco128.yaml",    # 데이터셋 설정 파일 경로
    imgsz=(640, 640),                   # 추론 이미지 크기 (height, width)
    conf_thres=0.25,                    # 객체 검출 신뢰도 임계값
    iou_thres=0.45,                     # NMS IoU 임계값
    max_det=1000,                       # 이미지당 최대 검출 수
    device="",                          # 연산 장치 (cuda:0, cpu 등)
    view_img=False,                     # 실시간 결과 화면 출력 여부
    save_txt=False,                     # 검출 결과를 txt 파일로 저장 여부
    save_format=0,                      # txt 저장 형식 (0: YOLO, 1: Pascal-VOC)
    save_csv=False,                     # 검출 결과를 CSV 파일로 저장 여부
    save_conf=False,                    # txt에 신뢰도 값 포함 여부
    save_crop=False,                    # 검출 영역 크롭 이미지 저장 여부
    nosave=False,                       # 결과 이미지/영상 저장 비활성화 여부
    classes=None,                       # 특정 클래스만 필터링 (예: [0, 1, 2])
    agnostic_nms=False,                 # 클래스 무관 NMS 적용 여부
    augment=False,                      # 추론 시 데이터 증강 적용 여부
    visualize=False,                    # 모델 내부 특징맵 시각화 여부
    update=False,                       # 모델 가중치 업데이트 여부
    project=ROOT / "runs/detect",       # 결과 저장 상위 디렉토리
    name="exp",                         # 결과 저장 하위 디렉토리 이름
    exist_ok=False,                     # 기존 디렉토리 덮어쓰기 허용 여부
    line_thickness=3,                   # 바운딩 박스 선 두께 (픽셀)
    hide_labels=False,                  # 라벨 텍스트 숨김 여부
    hide_conf=False,                    # 신뢰도 숫자 숨김 여부
    half=False,                         # FP16 반정밀도 추론 사용 여부
    dnn=False,                          # OpenCV DNN 백엔드 사용 여부 (ONNX 전용)
    vid_stride=1,                       # 영상 프레임 처리 간격
):
    # ── 입력 소스 유형 분류 ──────────────────────────────────────────────────
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    # ── 결과 저장 디렉토리 생성 ──────────────────────────────────────────────
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # ── 모델 로드 ────────────────────────────────────────────────────────────
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # ── 데이터 로더 설정 ─────────────────────────────────────────────────────
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # ── 모델 웜업 ────────────────────────────────────────────────────────────
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    # ── 프레임별 추론 루프 ───────────────────────────────────────────────────
    for path, im, im0s, vid_cap, s in dataset:

        # 전처리: numpy 배열을 텐서로 변환 및 정규화
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255  # 픽셀값 0~255 → 0.0~1.0 정규화
            if len(im.shape) == 3:
                im = im[None]  # 배치 차원 추가
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # 추론
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)

        # NMS(비최대 억제)로 중복 박스 제거
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # CSV 저장 경로 설정
        csv_path = save_dir / "predictions.csv"

        def write_to_csv(image_name, prediction, confidence):
            """검출 결과를 CSV 파일에 누적 저장하는 함수."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)

        # ── 검출 결과 처리 (이미지/프레임 단위) ─────────────────────────────
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            s += "{:g}x{:g} ".format(*im.shape[2:])
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 정규화 기준값 (whwh)
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # 번호판이 검출된 경우에만 처리
            if len(det):
                # 추론 크기 → 원본 이미지 크기로 좌표 역변환
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 클래스별 검출 수 로그 출력
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # ── 각 검출 박스 처리 ────────────────────────────────────────
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    # 번호판 영역 크롭 (여백 5픽셀 추가)
                    x1, y1, x2, y2 = map(int, xyxy)
                    pad = 5
                    x1 = max(x1 - pad, 0)
                    y1 = max(y1 - pad, 0)
                    x2 = min(x2 + pad, im0.shape[1])
                    y2 = min(y2 + pad, im0.shape[0])

                    plate_img = im0[y1:y2, x1:x2]
                    plate_text = ""

                    # ── OCR 텍스트 인식 ──────────────────────────────────────
                    if plate_img is not None and plate_img.size > 0:
                        try:
                            # 전처리: 확대 + 노이즈 제거 (컬러 유지)
                            processed = preprocess_plate(plate_img)

                            # PaddleOCR로 텍스트 인식
                            result = ocr.ocr(processed, cls=True)
                            LOGGER.info(f"OCR raw result: {result}")

                            # 신뢰도 0.4 이상인 인식 결과만 채택
                            if result and result[0]:
                                for line in result[0]:
                                    text, ocr_conf = line[1]
                                    if ocr_conf > 0.4:
                                        plate_text += text

                            if plate_text:
                                LOGGER.info(f"번호판 인식 결과: {plate_text}")

                        except Exception as e:
                            LOGGER.warning(f"OCR 오류: {e}")
                            plate_text = ""

                    # CSV 저장
                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    # TXT 저장 (YOLO 또는 Pascal-VOC 형식)
                    if save_txt:
                        if save_format == 0:
                            coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            )
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    # 바운딩 박스 그리기
                    if save_img or save_crop or view_img:
                        c = int(cls)
                        label = None if hide_labels else (f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    # 크롭 이미지 저장
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # ── 최종 프레임에 OCR 결과 텍스트 오버레이 ──────────────────────
            # annotator.result() 호출 이후에 텍스트를 그려야
            # 바운딩 박스에 덮어씌워지지 않고 위에 표시됨
            im0 = annotator.result()
            if plate_text:
                try:
                    # OpenCV BGR → PIL RGB 변환 후 한글 텍스트 렌더링
                    img_pil = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    font = ImageFont.truetype(font_path, 45)

                    # 번호판 박스 위쪽에 텍스트 표시
                    draw.text((x1, max(y1 - 70, 0)), plate_text, font=font, fill=(0, 255, 0))

                    # PIL RGB → OpenCV BGR 변환 후 복원
                    im0 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    LOGGER.warning(f"텍스트 렌더링 오류: {e}")

            # 실시간 화면 출력
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # ── 결과 이미지/영상 저장 ────────────────────────────────────────
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    # 영상의 경우 VideoWriter 초기화 후 프레임 단위로 기록
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # 프레임별 처리 속도 로그
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    # ── 전체 처리 속도 요약 출력 ─────────────────────────────────────────────
    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])


def parse_opt():
    """커맨드라인 인수를 파싱하는 함수."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="모델 가중치 경로")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="입력 소스 (파일/디렉토리/URL/웹캠)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="데이터셋 설정 파일 경로")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="추론 이미지 크기")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="객체 검출 신뢰도 임계값")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU 임계값")
    parser.add_argument("--max-det", type=int, default=1000, help="이미지당 최대 검출 수")
    parser.add_argument("--device", default="", help="연산 장치 (예: 0, cpu)")
    parser.add_argument("--view-img", action="store_true", help="실시간 결과 화면 출력")
    parser.add_argument("--save-txt", action="store_true", help="검출 결과 txt 저장")
    parser.add_argument("--save-format", type=int, default=0, help="txt 저장 형식 (0: YOLO, 1: Pascal-VOC)")
    parser.add_argument("--save-csv", action="store_true", help="검출 결과 CSV 저장")
    parser.add_argument("--save-conf", action="store_true", help="txt에 신뢰도 포함")
    parser.add_argument("--save-crop", action="store_true", help="검출 영역 크롭 이미지 저장")
    parser.add_argument("--nosave", action="store_true", help="결과 이미지/영상 저장 비활성화")
    parser.add_argument("--classes", nargs="+", type=int, help="특정 클래스만 필터링")
    parser.add_argument("--agnostic-nms", action="store_true", help="클래스 무관 NMS 적용")
    parser.add_argument("--augment", action="store_true", help="추론 시 데이터 증강 적용")
    parser.add_argument("--visualize", action="store_true", help="모델 특징맵 시각화")
    parser.add_argument("--update", action="store_true", help="모델 가중치 업데이트")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="결과 저장 상위 디렉토리")
    parser.add_argument("--name", default="exp", help="결과 저장 하위 디렉토리 이름")
    parser.add_argument("--exist-ok", action="store_true", help="기존 디렉토리 덮어쓰기 허용")
    parser.add_argument("--line-thickness", default=3, type=int, help="바운딩 박스 선 두께")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="라벨 숨김")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="신뢰도 숨김")
    parser.add_argument("--half", action="store_true", help="FP16 반정밀도 추론 사용")
    parser.add_argument("--dnn", action="store_true", help="OpenCV DNN 백엔드 사용 (ONNX 전용)")
    parser.add_argument("--vid-stride", type=int, default=1, help="영상 프레임 처리 간격")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 단일 값 입력 시 [h, w] 형태로 확장
    print_args(vars(opt))
    return opt


def main(opt):
    """의존성 확인 후 추론 실행."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)