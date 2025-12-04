# FastAPI + YOLOv8 서버용 Dockerfile
# Orange Pi RV2 (ARM64) 및 x86_64 지원

FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY backend/ ./backend/
COPY .env* ./ 2>/dev/null || true

# YOLO 모델 다운로드 (선택사항 - 빌드 시 다운로드)
# RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 포트 노출
EXPOSE 5000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5000/health || exit 1

# 서버 실행
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "5000"]