# HeadHunter 모델 설정 가이드

## 개요

HeadHunter는 CVPR 2021 논문 "Tracking Pedestrian Heads in Dense Crowd"에서 제안된 Head Detection 모델입니다.
탑뷰(항공 샷) 환경에서 사람의 머리를 감지하는 데 최적화되어 있습니다.

## 설치 방법

### 방법 1: pip 설치 (가능한 경우)

```bash
pip install head_detection
```

### 방법 2: GitHub에서 직접 설치

1. GitHub 저장소 클론:
```bash
git clone https://github.com/Sentient07/HeadHunter.git
cd HeadHunter
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 모델 가중치 다운로드:
   - 논문 페이지 또는 저장소에서 사전 학습된 가중치 다운로드
   - `weights/` 디렉토리에 저장

### 방법 3: 수동 설치

HeadHunter가 pip로 설치되지 않는 경우:

1. GitHub 저장소에서 코드 다운로드
2. `head_detection` 모듈을 Python 경로에 추가
3. 필요한 의존성 수동 설치

## 사용 방법

### 기본 사용

```bash
cd backend
python webcam_detection_headhunter.py
```

### 환경 변수 설정

```bash
# Confidence threshold 설정 (YOLO 대안 사용 시)
set CONFIDENCE_THRESHOLD=0.20

# PowerShell
$env:CONFIDENCE_THRESHOLD="0.20"
```

## 문제 해결

### 문제 1: head_detection 모듈을 찾을 수 없음

**해결 방법:**
- GitHub에서 HeadHunter 저장소를 직접 설치
- 또는 YOLOv8 기반 대안 사용 (코드에서 자동으로 전환)

### 문제 2: 모델 가중치를 찾을 수 없음

**해결 방법:**
1. 논문 페이지에서 사전 학습된 가중치 다운로드
2. 적절한 경로에 저장
3. 코드에서 모델 경로 수정

### 문제 3: CUDA 오류

**해결 방법:**
- CPU 모드로 실행 (일부 모델은 CPU 지원)
- 또는 CUDA/cuDNN 버전 확인

## 대안: YOLOv8 기반 Head Detection

HeadHunter가 설치되지 않은 경우, 코드는 자동으로 YOLOv8 Person Detection을 사용합니다.

**한계:**
- Person Detection이므로 머리만 보이는 경우 정확도가 낮을 수 있음
- Head Detection 전용 모델로 Fine-tuning 권장

**개선 방법:**
1. Brainwash + CroHD 데이터셋으로 YOLOv8 Fine-tuning
2. Head 클래스로 재학습
3. Fine-tuning된 모델 사용

## 참고 자료

- 논문: CVPR 2021 "Tracking Pedestrian Heads in Dense Crowd"
- GitHub: https://github.com/Sentient07/HeadHunter
- 데이터셋: CroHD (Head Tracking 21 챌린지)

## 다음 단계

1. HeadHunter 모델 설치 및 테스트
2. 탑뷰 영상으로 정확도 확인
3. 필요 시 Fine-tuning (실제 환경 데이터 추가)
