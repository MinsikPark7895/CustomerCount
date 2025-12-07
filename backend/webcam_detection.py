"""
노트북 카메라를 사용한 실시간 사람 인식
YOLOv8n 기반 Person Detection

사용법:
    python webcam_detection.py
"""

import cv2
from ultralytics import YOLO
import os
from datetime import datetime
import time
import torch

# PyTorch 2.6+ 호환성: torch.load의 weights_only 기본값을 False로 패치
# YOLO 모델은 신뢰할 수 있는 소스이므로 weights_only=False 사용
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """torch.load를 패치하여 weights_only=False를 기본값으로 설정"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# 모델 경로 설정
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")

def main():
    """메인 함수: 웹캠에서 실시간 사람 인식"""
    
    # 모델 경로 확인 및 절대 경로로 변환
    model_path = MODEL_PATH
    if not os.path.isabs(model_path):
        # 상대 경로인 경우 backend 디렉토리 기준으로 변환
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_full_path = os.path.join(script_dir, model_path)
        if os.path.exists(model_full_path):
            model_path = model_full_path
        elif os.path.exists(model_path):
            model_path = os.path.abspath(model_path)
    
    # 모델 로드
    print(f"모델 로딩 중: {model_path}")
    try:
        model = YOLO(model_path)
        print("모델 로드 완료!")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print(f"모델 파일이 존재하는지 확인하세요: {model_path}")
        return
    
    # 웹캠 초기화
    # 0은 기본 카메라, 다른 카메라를 사용하려면 1, 2 등으로 변경
    print("\n카메라 초기화 중...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
        # 다른 카메라 인덱스 시도
        print("다른 카메라 인덱스를 시도합니다...")
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"카메라 {i}를 성공적으로 열었습니다.")
                break
        else:
            print("사용 가능한 카메라를 찾을 수 없습니다.")
            return
    
    # 카메라 해상도 설정 (선택사항)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 실제 해상도 확인
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 카메라가 준비될 때까지 대기
    print("카메라 준비 중...")
    time.sleep(1)
    
    # 테스트 프레임 읽기
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        cap.release()
        return
    
    print(f"카메라 해상도: {actual_width}x{actual_height}")
    print("\n=== 웹캠 인식 시작 ===")
    print("종료하려면 'q' 키를 누르세요")
    print("-" * 40)
    
    # FPS 계산을 위한 변수
    fps_start_time = datetime.now()
    fps_frame_count = 0
    
    # OpenCV 창 생성 (Windows 호환성 개선)
    window_name = "Person Detection - Webcam"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_width, actual_height)
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("프레임을 읽을 수 없습니다.")
                break
            
            # 프레임이 비어있는지 확인
            if frame.size == 0:
                print("빈 프레임을 받았습니다.")
                continue
            
            # YOLO 추론 (person 클래스만)
            try:
                results = model(frame, verbose=False, classes=[0])[0]  # class 0 = person
            except Exception as e:
                print(f"YOLO 추론 오류: {e}")
                continue
            
            # Person 감지 결과 그리기
            person_count = 0
            for box in results.boxes:
                if int(box.cls) == 0:  # person class
                    person_count += 1
                    
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf)
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),  # 초록색
                        2
                    )
                    
                    # Confidence 점수 표시
                    label = f"Person {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1) - label_size[1] - 10),
                        (int(x1) + label_size[0], int(y1)),
                        (0, 255, 0),
                        -1
                    )
                    cv2.putText(
                        frame,
                        label,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                    )
            
            # 인원수 표시 (화면 상단)
            count_text = f"People Count: {person_count}"
            cv2.putText(
                frame,
                count_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )
            
            # FPS 계산 및 표시
            fps_frame_count += 1
            elapsed_time = (datetime.now() - fps_start_time).total_seconds()
            if elapsed_time > 0:
                fps = fps_frame_count / elapsed_time
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(
                    frame,
                    fps_text,
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # 모델 정보 표시
            model_text = f"Model: {os.path.basename(model_path)}"
            cv2.putText(
                frame,
                model_text,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            
            # 프레임 표시
            try:
                cv2.imshow(window_name, frame)
                
                # 창이 닫혔는지 확인
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n창이 닫혔습니다.")
                    break
            except cv2.error as e:
                print(f"화면 표시 오류: {e}")
                break
            
            # 'q' 키를 누르면 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 또는 ESC
                print("\n종료합니다...")
                break
                
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 리소스 해제
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("카메라가 종료되었습니다.")

if __name__ == "__main__":
    main()


