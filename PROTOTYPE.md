# 프로토타입 브랜치 가이드

## 개요

이 브랜치는 **프로토타입** 개발을 위한 것입니다.
- 목적: 빠른 개념 검증 및 테스트
- 모델: YOLOv8n (person detection)
- 상태: 개발 중

## ⚠️ 중요 사항

**이 브랜치는 프로토타입입니다.**
- 프로덕션 환경에서는 사용하지 마세요
- 정확도는 제한적일 수 있습니다
- 프로젝트가 구체화되면 Head Detection 또는 Fine-tuning 모델로 교체 예정

## 프로젝트 구조

```
CustomerCount/
├── backend/
│   ├── main.py              # FastAPI 서버 (프로토타입)
│   ├── config.py            # 설정 파일
│   └── README_PROTOTYPE.md  # 상세 가이드
├── docs/
│   └── Feature.md          # 기능 명세서
├── requirements.txt         # Python 의존성
├── .gitignore              # Git 제외 파일
├── README.md               # 프로젝트 메인 README
└── PROTOTYPE.md            # 이 파일
```

## 빠른 시작

### 1. 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

### 2. 실행

```bash
cd backend
python main.py
```

### 3. 테스트

브라우저에서 http://localhost:5000/docs 접속하여 API 테스트

## 현재 구현 상태

- ✅ FastAPI 서버 기본 구조
- ✅ YOLOv8n 모델 통합
- ✅ 이미지 업로드 API
- ✅ 인원수 카운팅
- ⏳ 데이터베이스 연동 (예정)
- ⏳ 프론트엔드 (예정)

## 프로토타입의 한계

### 1. 모델 한계
- YOLOv8n은 일반적인 각도(person detection)에 최적화
- 위에서 내려다보는 각도에서는 정확도 낮을 수 있음
- 예상 정확도: 50-70%

### 2. 기능 제한
- 데이터베이스 저장 미구현
- 프론트엔드 없음
- 추적(Tracking) 기능 없음

## 다음 단계 (프로덕션)

프로젝트가 구체화되면:

1. **모델 교체**
   - Head Detection 모델 적용
   - 또는 Fine-tuning 모델 사용

2. **기능 추가**
   - 데이터베이스 연동
   - 프론트엔드 개발
   - 추적 기능 추가

3. **최적화**
   - ONNX Runtime 적용
   - 성능 튜닝

## 참고 자료

- [backend/README_PROTOTYPE.md](backend/README_PROTOTYPE.md) - 상세 사용 가이드
- [docs/Feature.md](docs/Feature.md) - 전체 기능 명세서
- [README.md](README.md) - 프로젝트 메인 문서

