# 웹캠 실시간 사람 인식 가이드

## 개요

노트북 카메라를 사용하여 실시간으로 사람을 인식하는 기능입니다.
YOLOv8n 모델을 사용하여 카메라 화면에서 사람을 감지하고 인원수를 표시합니다.

## 파일 설명

1. **webcam_detection.py** - 기본 버전 (간단한 사용)
2. **webcam_detection_advanced.py** - 고급 버전 (설정 가능 옵션)

## 빠른 시작

### 기본 버전 실행

```bash
cd backend
python webcam_detection.py
```

### 고급 버전 실행

```bash
cd backend
python webcam_detection_advanced.py
```

## 고급 버전 옵션

```bash
# 다른 카메라 사용 (기본값: 0)
python webcam_detection_advanced.py --camera 1

# Confidence threshold 조정 (기본값: 0.25)
python webcam_detection_advanced.py --confidence 0.3

# 해상도 설정
python webcam_detection_advanced.py --width 1920 --height 1080

# 감지된 이미지 저장
python webcam_detection_advanced.py --save

# 모든 옵션 조합
python webcam_detection_advanced.py --camera 0 --confidence 0.3 --width 1280 --height 720 --save
```

## 사용 방법

1. **스크립트 실행**
   ```bash
   python webcam_detection.py
   ```

2. **카메라 화면 확인**
   - 웹캠 화면이 열립니다
   - 감지된 사람은 초록색 박스로 표시됩니다
   - 화면 상단에 인원수가 표시됩니다

3. **종료**
   - `q` 키를 누르면 종료됩니다
   - 또는 `Ctrl + C`로 중단할 수 있습니다

## 화면 정보

### 기본 버전
- **People Count**: 현재 프레임에서 감지된 인원수
- **FPS**: 초당 프레임 수 (성능 지표)
- **Model**: 사용 중인 모델 이름

### 고급 버전
- **People**: 현재 프레임에서 감지된 인원수
- **Avg**: 평균 인원수
- **Max**: 최대 인원수
- **FPS**: 초당 프레임 수
- **Conf**: Confidence threshold 값
- **Model**: 사용 중인 모델 이름

## 문제 해결

### 카메라가 열리지 않는 경우

```bash
# 다른 카메라 인덱스 시도
python webcam_detection_advanced.py --camera 1
python webcam_detection_advanced.py --camera 2
```

### 성능이 느린 경우

1. **해상도 낮추기**
   ```bash
   python webcam_detection_advanced.py --width 640 --height 480
   ```

2. **Confidence threshold 높이기** (더 확실한 감지만)
   ```bash
   python webcam_detection_advanced.py --confidence 0.5
   ```

### 모델 로드 실패

- 인터넷 연결 확인 (첫 실행 시 모델 다운로드 필요)
- `yolov8n.pt` 파일이 `~/.ultralytics/weights/` 경로에 있는지 확인

### 화면이 보이지 않는 경우

- 다른 애플리케이션에서 카메라를 사용 중인지 확인
- 카메라 권한 확인 (Windows: 설정 > 개인 정보 > 카메라)

## 성능 최적화

### CPU 환경
- 해상도: 640x480 권장
- FPS: 약 5-10 FPS 예상

### GPU 환경 (CUDA)
- 해상도: 1280x720 가능
- FPS: 약 20-30 FPS 예상

## 다음 단계

프로젝트가 구체화되면:
1. Head Detection 모델로 교체
2. Fine-tuning 모델 적용
3. 추적(Tracking) 기능 추가
4. 데이터베이스 연동

## 참고

- 프로토타입 버전이므로 정확도는 제한적일 수 있습니다
- 위에서 내려다보는 각도에서는 성능이 낮을 수 있습니다
- 프로덕션 환경에서는 Head Detection 모델 사용 권장


