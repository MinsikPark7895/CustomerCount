"""
자동 인원 파악 시스템 - 프로토타입
YOLOv8n 기반 Person Detection API 서버

주의: 이 코드는 프로토타입입니다.
프로덕션 환경에서는 Head Detection 모델 또는 Fine-tuning 모델로 교체 예정.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
from typing import Optional
import os
from pathlib import Path
import torch

# PyTorch 2.6+ 호환성: torch.load의 weights_only 기본값을 False로 패치
# YOLO 모델은 신뢰할 수 있는 소스이므로 weights_only=False 사용
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """torch.load를 패치하여 weights_only=False를 기본값으로 설정"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# FastAPI 앱 초기화
app = FastAPI(
    title="Customer Count API - Prototype",
    description="YOLOv8n 기반 인원 카운팅 프로토타입",
    version="0.1.0"
)

# 모델 로드 (프로토타입: YOLOv8n person detection)
# TODO: 프로덕션에서는 Head Detection 모델로 교체 예정
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
model = None

def load_model():
    """모델 로드 함수"""
    global model
    try:
        print(f"모델 로딩 중: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("모델 로드 완료")
        return model
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        raise

# 앱 시작 시 모델 로드
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    """헬스 체크 엔드포인트"""
    return {
        "status": "running",
        "model": MODEL_PATH,
        "version": "0.1.0",
        "note": "프로토타입 - YOLOv8n person detection"
    }

@app.get("/health")
async def health_check():
    """상세 헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    이미지 업로드 및 인원수 감지
    
    Args:
        file: JPEG 이미지 파일
        
    Returns:
        {
            "count": int,           # 감지된 인원수
            "timestamp": str,       # 처리 시점
            "model": str,           # 사용된 모델
            "confidence_threshold": float  # 사용된 confidence threshold
        }
    """
    if model is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    
    # 파일 타입 검증
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다")
        
        # YOLO 추론 (person 클래스만 필터링)
        # COCO 데이터셋에서 person 클래스는 0번
        results = model(img, verbose=False)[0]
        
        # Person 클래스만 필터링 (class 0)
        # Confidence threshold는 기본값 사용 (0.25)
        person_boxes = [
            box for box in results.boxes 
            if int(box.cls) == 0  # person class
        ]
        
        person_count = len(person_boxes)
        
        # 결과 반환
        return JSONResponse({
            "count": person_count,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_PATH,
            "confidence_threshold": 0.25,
            "image_size": {
                "width": img.shape[1],
                "height": img.shape[0]
            },
            "detections": len(results.boxes),  # 전체 감지 수 (디버깅용)
            "note": "프로토타입 - 정확도는 제한적일 수 있습니다"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")

@app.post("/upload/advanced")
async def upload_image_advanced(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.25,
    iou: Optional[float] = 0.45
):
    """
    고급 이미지 업로드 및 인원수 감지 (파라미터 조정 가능)
    
    Args:
        file: JPEG 이미지 파일
        confidence: Confidence threshold (기본값: 0.25)
        iou: IoU threshold for NMS (기본값: 0.45)
        
    Returns:
        {
            "count": int,
            "timestamp": str,
            "model": str,
            "confidence_threshold": float,
            "detections": list  # 각 감지 정보
        }
    """
    if model is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다")
        
        # YOLO 추론 (커스텀 threshold)
        results = model(
            img, 
            conf=confidence,
            iou=iou,
            verbose=False
        )[0]
        
        # Person 클래스만 필터링
        person_detections = []
        for box in results.boxes:
            if int(box.cls) == 0:  # person class
                person_detections.append({
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                })
        
        person_count = len(person_detections)
        
        return JSONResponse({
            "count": person_count,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_PATH,
            "confidence_threshold": confidence,
            "iou_threshold": iou,
            "detections": person_detections,
            "image_size": {
                "width": img.shape[1],
                "height": img.shape[0]
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

