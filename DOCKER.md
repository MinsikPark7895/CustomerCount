# Docker 사용 가이드

## 빠른 시작

### 1. Docker Compose로 실행 (권장)

```bash
# 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f api

# 중지
docker-compose down
```

서버 접속: http://localhost:5000/docs

### 2. Dockerfile만 사용

```bash
# 이미지 빌드
docker build -t customercount:latest .

# 컨테이너 실행
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/backend:/app/backend \
  --name customercount-api \
  customercount:latest

# 로그 확인
docker logs -f customercount-api

# 중지 및 제거
docker stop customercount-api
docker rm customercount-api
```

---

## Orange Pi RV2 (ARM64) 배포

### 방법 1: 직접 빌드 (권장)

Orange Pi에서 직접 실행:

```bash
# Docker 설치 확인
docker --version
docker-compose --version

# 프로젝트 클론 또는 복사
cd /path/to/CustomerCount

# 빌드 및 실행
docker-compose up -d --build
```

### 방법 2: 멀티 아키텍처 이미지

개발 PC에서 ARM64 이미지 빌드:

```bash
# Buildx 설정 (최초 1회)
docker buildx create --use --name multi-arch-builder

# ARM64 이미지 빌드
docker buildx build \
  --platform linux/arm64 \
  -t customercount:arm64 \
  --load \
  .

# 이미지 저장
docker save customercount:arm64 | gzip > customercount-arm64.tar.gz

# Orange Pi로 전송
scp customercount-arm64.tar.gz user@orange-pi-ip:/tmp/

# Orange Pi에서 로드
docker load < /tmp/customercount-arm64.tar.gz
docker-compose up -d
```

---

## 환경 변수 설정

`.env` 파일 생성:

```bash
# 서버 설정
API_PORT=5000
YOLO_MODEL=yolov8n.pt

# 데이터베이스 (선택)
DATABASE_URL=postgresql://user:password@db:5432/customercount

# 개발 모드
DEBUG=false
```

---

## 볼륨 관리

### YOLO 모델 캐시

모델 파일은 `yolo_models` 볼륨에 캐시됩니다:

```bash
# 볼륨 확인
docker volume ls

# 볼륨 상세 정보
docker volume inspect customercount_yolo_models

# 볼륨 삭제 (모델 재다운로드)
docker-compose down -v
```

### 업로드 이미지 저장

```bash
# 호스트에 디렉토리 생성
mkdir -p data/uploads

# docker-compose.yml에 이미 설정됨
# - ./data/uploads:/app/uploads
```

---

## 프로덕션 배포

### 1. docker-compose.prod.yml 생성

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: customercount-api
    ports:
      - "5000:5000"
    volumes:
      - yolo_models:/root/.cache/ultralytics
      - ./data/uploads:/app/uploads
      # 개발용 코드 마운트 제거
    environment:
      - PYTHONUNBUFFERED=1
      - DEBUG=false
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G

volumes:
  yolo_models:
```

### 2. 실행

```bash
docker-compose -f docker-compose.prod.yml up -d --build
```

---

## 데이터베이스 추가

PostgreSQL을 사용하려면 `docker-compose.yml`에서 주석 해제:

```yaml
services:
  db:
    image: postgres:15-alpine
    container_name: customercount-db
    environment:
      - POSTGRES_USER=customercount
      - POSTGRES_PASSWORD=changeme
      - POSTGRES_DB=customercount
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  postgres_data:
```

---

## 트러블슈팅

### 1. 모델 다운로드 실패

```bash
# 컨테이너 내부에서 수동 다운로드
docker exec -it customercount-api python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 2. 메모리 부족 (Orange Pi)

```yaml
# docker-compose.yml에 리소스 제한 추가
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 1.5G
```

### 3. 권한 오류

```bash
# 볼륨 권한 수정
sudo chown -R 1000:1000 data/
```

### 4. 포트 충돌

```yaml
# docker-compose.yml에서 포트 변경
ports:
  - "8000:5000"  # 호스트:컨테이너
```

---

## 로그 및 모니터링

```bash
# 실시간 로그
docker-compose logs -f api

# 최근 100줄
docker-compose logs --tail=100 api

# 컨테이너 상태
docker-compose ps

# 리소스 사용량
docker stats customercount-api

# 헬스체크 확인
docker inspect --format='{{.State.Health.Status}}' customercount-api
```

---

## 업데이트 및 재배포

```bash
# 코드 업데이트 후
docker-compose down
docker-compose up -d --build

# 또는 무중단 재시작 (rolling update)
docker-compose up -d --no-deps --build api
```

---

## 개발 워크플로우

```bash
# 개발 모드 실행 (코드 자동 반영)
docker-compose up

# 다른 터미널에서 코드 수정
# uvicorn이 자동으로 재시작됨

# 테스트
curl -X POST http://localhost:5000/upload \
  -F "file=@test_image.jpg"
```

---

## 정리

```bash
# 컨테이너 중지 및 제거
docker-compose down

# 볼륨까지 제거 (데이터 삭제 주의!)
docker-compose down -v

# 이미지 제거
docker rmi customercount:latest

# 모든 미사용 리소스 정리
docker system prune -a --volumes
```