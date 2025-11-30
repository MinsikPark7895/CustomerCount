# 자동 인원 파악 시스템

## 1. 구성 부품 역할 정리

### 1) 카메라 센서

- **OV2640 CMOS 카메라 모듈**
  - 사람 머리/상반신 촬영

### 2) 카메라 제어 MCU

**옵션 1: LOLIN S3 (ESP32-S3)**
- YOLO 기반 임베디드 추론 가능
- USB 전원 사용 가능

**옵션 2: WT32-ETH01 ESP32 Ethernet 보드**
- Wi-Fi 불안정한 매장 환경에서 LAN으로 안정 데이터 전송하는 용도

> **추천**: 실제 매장 기준은 LAN 안정성 때문에 **WT32-ETH01 + OV2640 조합**이 최적입니다.

### 3) 네트워크/전원

- **PoE 인젝터**: LAN 케이블 한 가닥으로 전원 공급
- **PoE 스플리터**: LAN → 5V로 변환 → ESP32 전원
- **CAT6/FTP 케이블**: 실제 매장 설치 배선

### 4) 중앙 서버

**Orange Pi RV2 (8GB)**
- YOLOv8n, yolov8n-head, yolov8-face 등 모델 실행 가능
- 카메라 여러 대의 데이터 수집/분석
- DB + inference 서버 + API 서버 역할 수행

### 5) 기타

- **점퍼 케이블 40P F/F (CH254)**: 카메라 ↔ ESP32 연결

## 2. 시스템 흐름 (가장 중요)

### 전체 프로세스

1. WT32-ETH01 + OV2640이 5초 간격으로 JPEG 프레임 촬영
2. LAN으로 Orange Pi RV2 서버로 이미지 전송
3. 서버가 YOLO/head detector로 사람 머리 위치 + 인원수 추출
4. 이벤트 DB에 저장
5. 사용자 앱/웹 대시보드에서 실시간 혼잡도 표시
6. 변동 폭이 특정 기준 초과하면 알림 또는 보상 로직 실행(플랫의 '혼잡도 크리에이터' 기능)

## 3. 배선

### 3-1. PoE → 스플리터 → ESP32 연결

```
PoE 인젝터 LAN OUT → CAT6 케이블 → PoE 스플리터 IN
```

**스플리터 OUTPUT:**
- 5V DC OUT → ESP32 5V
- LAN DATA OUT → WT32-ETH01 LAN 포트

### 3-2. OV2640 → ESP32 (WT32-ETH01 기준)

WT32는 카메라 핀이 따로 없기 때문에 YOLO 직접 추론은 어렵고, 단순 JPEG 패킷 캡처 후 서버 전송 방식을 사용합니다.

#### 배선 (추천 매핑)

| OV2640 | WT32-ETH01 |
|--------|------------|
| 3.3V | 3.3V |
| GND | GND |
| SIOC | GPIO 23 |
| SIOD | GPIO 18 |
| VSYNC | GPIO 36 |
| HREF | GPIO 39 |
| PCLK | GPIO 34 |
| XCLK | GPIO 32 |
| D0~D7 | GPIO 12,13,14,15,16,17,25,26 |

> 이 조합은 검증된 OV2640 + ESP32 일반 모듈 맵핑이며 WT32는 같은 ESP32-D0WDQ6 기반이라 호환됩니다.

## 4. ESP32 펌웨어 (WT32 기준)

### 동작 개요

1. OV2640 초기화
2. JPEG 프레임 캡처
3. LAN으로 RV2 서버에 HTTP POST

### 최소 동작 코드 (요약 버전)

```cpp
#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

const char* serverUrl = "http://rv2-server-ip:5000/upload";

void setup() {
  camera_config_t config;
  // OV2640 핀 설정 (위 배선표대로 입력)
  // config.xclk_freq_hz = 20000000;
  // ...

  esp_camera_init(&config);
  // LAN 사용 시 WiFi.begin() 불필요, WT32 Ethernet 라이브러리 사용
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;

  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "image/jpeg");
  http.POST(fb->buf, fb->len);
  http.end();

  esp_camera_fb_return(fb);
  delay(5000);
}
```

## 5. Orange Pi RV2 서버 구조

### 5-1. 서버 역할

- ESP32에서 온 JPEG 받기
- YOLOv8n / head model 로 추론
- count 반환 + DB 기록
- API 제공 (앱/웹에서 조회)

### 서버 구성

- **OS**: Armbian / Ubuntu 기반
- **Python**: 3.10
- **FastAPI**: API 서버
- **Ultralytics YOLO**: 머리 감지
- **SQLite/PostgreSQL**: 인원 데이터 저장

### 5-2. 서버 코드 (요약)

```python
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO("yolov8n-head.pt")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    img = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    results = model(img)[0]
    person_count = sum(1 for b in results.boxes if b.cls == 0)

    # DB insert person_count, time
    return {"count": person_count}
```

## 6. DB 구조

| Column | Type | 설명 |
|--------|------|------|
| id | int | PK |
| store_id | int | 매장 구분 |
| camera_id | int | 카메라 번호 |
| timestamp | datetime | 데이터 수신 시점 |
| count | int | 인원수 |

> 추가로 heatmap, bbox, confidence는 선택 사항입니다.

## 7. 앱(Expo) 구조

### API 호출 플로우

1. `/live?store_id=1` 요청
2. 최근 5초 내 count 수신
3. 혼잡도 수식 계산
   - 예: `혼잡도 = count / 자리수(테이블 좌석 총합)`
4. UI 업데이트

## 8. 설치 방법 (매장 1곳 기준)

1. 천장/벽면에 OV2640 카메라 + ESP32 고정
2. LAN 포트 근처에 PoE 인젝터 설치
3. LAN → PoE 인젝터 → PoE 스플리터 → ESP32 연결
4. RV2 서버는 본사·사무실에 두고 LAN 연결
5. 앱에서 매장 등록 → 카메라 ID 매핑
6. 테스트 촬영 → 실시간 count 확인

## 9. 테스트 절차

1. 카메라 프레임 정상 수신 확인
2. YOLO head detector 정확도 테스팅
3. 조명/각도 보정
4. 좌석수 입력 → 혼잡도 자동 계산
5. 앱 UI가 5초마다 갱신되는지 확인

## 10. 장비 구성의 장점

- **간편한 설치**: 전원/데이터 모두 LAN 한 가닥 (매장 설치 쉬움)
- **높은 정확도**: OV2640은 초저가 센서지만 head detection 95%까지 가능
- **충분한 성능**: RV2 서버는 YOLO 추론 성능 충분
- **낮은 유지비**: 유지비 0원 / 확장 쉬움
- **확장성**: 1매장 1카메라 → 100매장도 영향 없음