# HeadHunter 설치 가이드

## 현재 상태

현재 HeadHunter가 **설치되지 않아** YOLOv8 대안으로 실행되고 있습니다.
터미널에서 다음과 같은 메시지가 나타납니다:

```
경고: head_detection 모듈을 찾을 수 없습니다.
대안으로 YOLOv8 기반 Head Detection을 사용합니다...
```

## 설치 방법 (GitHub에서 직접 설치)

### 단계별 설치 가이드

#### 1단계: 저장소 클론

```powershell
# 프로젝트 루트로 이동
cd "C:\Users\pms19\OneDrive\바탕 화면\CustomerCount"

# HeadHunter 저장소 클론
git clone https://github.com/Sentient07/HeadHunter.git

# HeadHunter 디렉토리로 이동
cd HeadHunter
```

또는 GitHub에서 ZIP 파일을 다운로드하여 압축 해제할 수도 있습니다.

#### 2단계: 가상환경 활성화 (현재 프로젝트 가상환경 사용)

```powershell
# 프로젝트의 가상환경으로 돌아가기
cd "C:\Users\pms19\OneDrive\바탕 화면\CustomerCount"

# 가상환경 활성화 (이미 활성화되어 있다면 생략)
.\venv\Scripts\Activate.ps1
```

#### 3단계: HeadHunter 의존성 설치

```powershell
# HeadHunter 디렉토리로 이동
cd HeadHunter

# 의존성 설치 (requirements.txt 사용)
pip install -r requirements.txt
```

**주의:** HeadHunter의 requirements.txt가 현재 프로젝트와 충돌할 수 있습니다.
충돌이 발생하면 의존성 버전을 조정하거나 별도 가상환경을 사용하세요.

#### 4단계: HeadHunter 패키지 설치

```powershell
# HeadHunter 디렉토리에서 실행
pip install .
```

또는 개발 모드로 설치 (코드 수정 시 즉시 반영):
```powershell
pip install -e .
```

#### 5단계: 모델 가중치 준비

HeadHunter는 사전 학습된 모델 가중치가 필요합니다:

1. **논문 페이지 확인:**
   - CVPR 2021 논문 페이지에서 가중치 다운로드 링크 확인
   - 또는 GitHub Issues에서 가중치 다운로드 링크 확인

2. **가중치 저장 위치:**
   - `HeadHunter/weights/` 디렉토리에 저장
   - 또는 코드에서 지정한 경로에 저장

3. **가중치가 없는 경우:**
   - 직접 학습 필요 (시간이 많이 소요됨)
   - 또는 다른 사전 학습된 모델 사용

## 설치 확인

설치가 완료되었는지 확인:

```powershell
python -c "from head_detection import HeadDetector; print('HeadHunter 설치 완료!')"
```

성공하면 "HeadHunter 설치 완료!" 메시지가 나타납니다.

## 시스템 요구사항

HeadHunter는 다음이 필요합니다:
- **Nvidia Driver**: >= 418
- **CUDA**: 10.0 및 호환 cuDNN
- **Python**: 호환 버전

**주의:** CUDA가 없으면 CPU 모드로 실행되지만 성능이 매우 저하됩니다.

## 설치 후 사용

설치가 완료되면:

```powershell
cd "C:\Users\pms19\OneDrive\바탕 화면\CustomerCount\backend"
python webcam_detection_headhunter.py
```

이제 HeadHunter 모델이 자동으로 로드됩니다.

## 문제 해결

### 문제 1: CUDA 오류

**증상:**
```
CUDA error: no kernel image is available
```

**해결 방법:**
- CUDA 10.0 설치 (HeadHunter 요구사항)
- 또는 CPU 모드로 실행 (매우 느림)
- 또는 더 최신 CUDA 버전 사용 (코드 수정 필요할 수 있음)

### 문제 2: 의존성 충돌

**증상:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages
```

**해결 방법:**
1. **별도 가상환경 사용 (권장):**
   ```powershell
   conda create -n headhunter python=3.8
   conda activate headhunter
   cd HeadHunter
   pip install -r requirements.txt
   pip install .
   ```

2. **의존성 버전 조정:**
   - requirements.txt의 버전을 현재 프로젝트와 호환되도록 수정

### 문제 3: 모델 가중치 없음

**증상:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'weights/...'
```

**해결 방법:**
1. 논문 페이지나 GitHub에서 가중치 다운로드
2. `weights/` 디렉토리에 저장
3. 코드에서 가중치 경로 확인 및 수정

### 문제 4: import 오류

**증상:**
```
ModuleNotFoundError: No module named 'head_detection'
```

**해결 방법:**
1. `pip install .` 명령이 성공했는지 확인
2. 가상환경이 올바르게 활성화되었는지 확인
3. Python 경로 확인:
   ```powershell
   python -c "import sys; print(sys.path)"
   ```

## 참고 자료

- **GitHub 저장소:** https://github.com/Sentient07/HeadHunter
- **논문:** CVPR 2021 "Tracking Pedestrian Heads in Dense Crowd"
- **설치 관련 Issues:** GitHub Issues에서 설치 관련 도움 확인

## 빠른 설치 스크립트 (PowerShell)

```powershell
# 프로젝트 루트로 이동
cd "C:\Users\pms19\OneDrive\바탕 화면\CustomerCount"

# HeadHunter 클론
git clone https://github.com/Sentient07/HeadHunter.git

# 가상환경 활성화
.\venv\Scripts\Activate.ps1

# HeadHunter 설치
cd HeadHunter
pip install -r requirements.txt
pip install .

# 설치 확인
python -c "from head_detection import HeadDetector; print('설치 완료!')"

# 프로젝트로 돌아가기
cd ..\backend
```
