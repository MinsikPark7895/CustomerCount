# HeadHunter 실행 가이드

## 현재 상황

- `webcam_detection_headhunter.py` 파일이 `backend/` 디렉토리에 있음
- HeadHunter-master가 프로젝트 루트에 있음
- backend와 HeadHunter-master에 각각 별도의 venv가 있음

## 실행 방법

### 방법 1: 현재 프로젝트 venv 사용 (권장)

#### 1단계: 프로젝트 venv 활성화

```powershell
# 프로젝트 루트로 이동
cd "C:\Users\pms19\OneDrive\바탕 화면\CustomerCount"

# venv 활성화
.\venv\Scripts\Activate.ps1
```

#### 2단계: HeadHunter 설치 (필요한 경우)

HeadHunter가 설치되지 않았다면:

```powershell
# HeadHunter-master로 이동
cd HeadHunter-master

# HeadHunter 설치
pip install -e .

# 프로젝트로 돌아가기
cd ..
```

**주의:** HeadHunter의 의존성이 현재 프로젝트와 충돌할 수 있습니다.

#### 3단계: 스크립트 실행

```powershell
# backend 디렉토리로 이동
cd backend

# 스크립트 실행
python webcam_detection_headhunter.py
```

### 방법 2: YOLOv8 대안 사용 (HeadHunter 없이)

HeadHunter가 설치되지 않아도 YOLOv8 대안으로 자동 실행됩니다:

```powershell
# 프로젝트 루트로 이동
cd "C:\Users\pms19\OneDrive\바탕 화면\CustomerCount"

# venv 활성화
.\venv\Scripts\Activate.ps1

# backend로 이동
cd backend

# 실행 (HeadHunter 없으면 자동으로 YOLOv8 사용)
python webcam_detection_headhunter.py
```

## 실행 시 나타나는 메시지

### HeadHunter가 설치된 경우

```
HeadHunter 모듈을 찾았습니다!
HeadHunter 모델 로딩 중...
HeadHunter 모델 로드 완료! (가중치: ...)
```

### HeadHunter가 설치되지 않은 경우

```
경고: head_detection 모듈을 찾을 수 없습니다.
대안으로 YOLOv8 기반 Head Detection을 사용합니다...

YOLOv8 모델 로딩 중...
YOLOv8 모델 로드 완료!
```

## 문제 해결

### 문제 1: 모듈을 찾을 수 없음

**증상:**
```
ModuleNotFoundError: No module named 'head_detection'
```

**해결:**
1. HeadHunter-master에서 `pip install -e .` 실행
2. 또는 YOLOv8 대안 사용 (자동으로 전환됨)

### 문제 2: 모델 가중치를 찾을 수 없음

**증상:**
```
경고: HeadHunter 모델 가중치를 찾을 수 없습니다.
```

**해결:**
1. 환경 변수 설정:
   ```powershell
   $env:HEADHUNTER_WEIGHTS="C:\path\to\weights\model.pth"
   ```
2. 또는 가중치 파일을 `HeadHunter-master/weights/` 디렉토리에 저장
3. 또는 YOLOv8 대안 사용

### 문제 3: CUDA 오류

**증상:**
```
CUDA error: ...
```

**해결:**
- CPU 모드로 실행 (매우 느림)
- 또는 YOLOv8 대안 사용 (CUDA 문제 없음)

## 환경 변수 설정 (선택사항)

### HeadHunter 가중치 경로

```powershell
$env:HEADHUNTER_WEIGHTS="C:\Users\pms19\OneDrive\바탕 화면\CustomerCount\HeadHunter-master\weights\model.pth"
```

### HeadHunter Confidence Threshold

```powershell
$env:HEADHUNTER_CONF_THRESH="0.3"
```

### YOLOv8 Confidence Threshold (대안 사용 시)

```powershell
$env:CONFIDENCE_THRESHOLD="0.20"
```

## 빠른 실행 스크립트

PowerShell에서 한 번에 실행:

```powershell
# 프로젝트 루트로 이동
cd "C:\Users\pms19\OneDrive\바탕 화면\CustomerCount"

# venv 활성화
.\venv\Scripts\Activate.ps1

# backend로 이동 및 실행
cd backend
python webcam_detection_headhunter.py
```

## 사용 방법

1. **스크립트 실행**
   - 위 명령으로 실행

2. **카메라 화면 확인**
   - 웹캠 화면이 열립니다
   - 감지된 머리는 초록색 박스로 표시됩니다
   - 화면 상단에 인원수가 표시됩니다

3. **종료**
   - `q` 키를 누르면 종료됩니다
   - 또는 `Ctrl + C`로 중단할 수 있습니다

## 참고

- HeadHunter가 없어도 YOLOv8 대안으로 자동 실행됩니다
- YOLOv8은 Person Detection이므로 머리만 보이는 경우 정확도가 낮을 수 있습니다
- HeadHunter를 사용하려면 모델 가중치 파일이 필요합니다
