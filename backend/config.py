"""
설정 파일
프로토타입용 기본 설정
"""

import os
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# 모델 설정
MODEL_CONFIG = {
    # 프로토타입: YOLOv8n person detection
    "model_path": os.getenv("MODEL_PATH", "yolov8n.pt"),
    
    # 프로덕션 예정 모델 (주석 처리)
    # "model_path": "yolov8n-head.pt",  # Head Detection 모델
    # "model_path": "models/custom-head-detection.pt",  # Fine-tuning 모델
    
    # 추론 설정
    "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.25")),
    "iou_threshold": float(os.getenv("IOU_THRESHOLD", "0.45")),
    
    # 이미지 설정
    "image_size": int(os.getenv("IMAGE_SIZE", "640")),  # YOLOv8 기본값
}

# 서버 설정
SERVER_CONFIG = {
    "host": os.getenv("HOST", "0.0.0.0"),
    "port": int(os.getenv("PORT", "5000")),
    "reload": os.getenv("RELOAD", "false").lower() == "true",
}

# 데이터베이스 설정 (프로토타입에서는 사용 안 함)
# TODO: 프로덕션에서 DB 연동 시 사용
DATABASE_CONFIG = {
    "type": os.getenv("DB_TYPE", "sqlite"),  # sqlite, postgresql
    "path": os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "customer_count.db")),
}

# 로깅 설정
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

