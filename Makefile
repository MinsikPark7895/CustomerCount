# CustomerCount - Docker 명령어 단축키

.PHONY: help build up down logs restart clean test health

help: ## 도움말 표시
	@echo "CustomerCount Docker 명령어"
	@echo ""
	@echo "사용법: make [명령어]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Docker 이미지 빌드
	docker-compose build

up: ## 서버 시작 (백그라운드)
	docker-compose up -d
	@echo ""
	@echo "서버 시작됨: http://localhost:5000/docs"

dev: ## 개발 모드 실행 (로그 출력)
	docker-compose up

down: ## 서버 중지
	docker-compose down

logs: ## 로그 확인 (실시간)
	docker-compose logs -f api

restart: ## 서버 재시작
	docker-compose restart api

rebuild: ## 재빌드 후 시작
	docker-compose down
	docker-compose up -d --build
	@echo ""
	@echo "서버 재시작됨: http://localhost:5000/docs"

clean: ## 컨테이너 및 볼륨 제거
	docker-compose down -v

ps: ## 실행 중인 컨테이너 확인
	docker-compose ps

health: ## 헬스체크
	@curl -s http://localhost:5000/health || echo "서버가 실행중이지 않습니다"

test: ## API 테스트 (테스트 이미지 필요)
	@if [ -f "test_image.jpg" ]; then \
		curl -X POST http://localhost:5000/upload -F "file=@test_image.jpg"; \
	else \
		echo "test_image.jpg 파일이 없습니다"; \
	fi

shell: ## 컨테이너 쉘 접속
	docker exec -it customercount-api /bin/bash

stats: ## 리소스 사용량 확인
	docker stats customercount-api --no-stream

prune: ## 미사용 Docker 리소스 정리
	docker system prune -f

# Orange Pi 배포
deploy-arm: ## ARM64 이미지 빌드 (Orange Pi용)
	docker buildx build --platform linux/arm64 -t customercount:arm64 --load .

save-arm: ## ARM64 이미지 저장
	docker save customercount:arm64 | gzip > customercount-arm64.tar.gz
	@echo "이미지 저장됨: customercount-arm64.tar.gz"