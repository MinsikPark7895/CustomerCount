# 프로토타입 - YOLOv8n Person Detection

## 개요

이 디렉토리는 **프로토타입** 코드입니다.
- 모델: YOLOv8n (person detection)
- 목적: 빠른 개념 검증 및 테스트
- **주의**: 프로덕션에서는 Head Detection 모델 또는 Fine-tuning 모델로 교체 예정

## 설치

```bash
# 가상 환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 실행

```bash
# 서버 시작
cd backend
python main.py

# 또는 uvicorn 직접 사용
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

서버가 시작되면:
- API 문서: http://localhost:5000/docs
- 헬스 체크: http://localhost:5000/health

## API 사용법

### 1. 기본 이미지 업로드

```bash
curl -X POST "http://localhost:5000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### 2. 고급 옵션 (confidence threshold 조정)

```bash
curl -X POST "http://localhost:5000/upload/advanced?confidence=0.3&iou=0.5" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### 3. Python 예제

```python
import requests

url = "http://localhost:5000/upload"
files = {"file": open("test_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## 응답 형식

```json
{
  "count": 5,
  "timestamp": "2024-01-01T12:00:00",
  "model": "yolov8n.pt",
  "confidence_threshold": 0.25,
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "detections": 5,
  "note": "프로토타입 - 정확도는 제한적일 수 있습니다"
}
```

## 주의사항

1. **프로토타입 한계**
   - YOLOv8n은 일반적인 각도(person detection)에 최적화됨
   - 위에서 내려다보는 각도에서는 정확도가 낮을 수 있음
   - 테스트 및 개념 검증 용도로만 사용

2. **모델 자동 다운로드**
   - 첫 실행 시 `yolov8n.pt` 모델이 자동으로 다운로드됨 (약 6MB)
   - 다운로드 위치: `~/.ultralytics/weights/`

3. **성능**
   - CPU: 약 100-200ms/이미지
   - GPU: 약 10-20ms/이미지

## 다음 단계

프로젝트가 구체화되면:
1. Head Detection 모델로 교체
2. Fine-tuning 모델 적용
3. 데이터베이스 연동
4. 프론트엔드 개발

## 문제 해결

### 모델 로드 실패
- 인터넷 연결 확인 (첫 다운로드 필요)
- 디스크 공간 확인

### 낮은 정확도
- 프로토타입의 한계 (예상됨)
- Head Detection 모델 또는 Fine-tuning 고려

### 메모리 부족
- 이미지 크기 줄이기
- 배치 크기 조정

