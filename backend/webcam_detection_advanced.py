"""
노트북 카메라를 사용한 실시간 사람 인식 (고급 버전)
YOLOv8n 기반 Person Detection with 설정 가능 옵션

사용법:
    python webcam_detection_advanced.py
    python webcam_detection_advanced.py --camera 1 --confidence 0.3
"""

import cv2
from ultralytics import YOLO
import os
import argparse
from datetime import datetime
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

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="웹캠 실시간 사람 인식")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="모델 경로 (기본값: yolov8n.pt)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="카메라 인덱스 (기본값: 0)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold (기본값: 0.25)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="카메라 해상도 너비 (기본값: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="카메라 해상도 높이 (기본값: 720)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="감지된 이미지 저장 (선택사항)"
    )
    return parser.parse_args()

def main():
    """메인 함수: 웹캠에서 실시간 사람 인식 (고급)"""
    
    args = parse_args()
    
    # 모델 로드
    print(f"모델 로딩 중: {args.model}")
    try:
        model = YOLO(args.model)
        print("모델 로드 완료!")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"카메라 {args.camera}를 열 수 없습니다.")
        print("다른 카메라를 사용하려면 --camera 옵션을 변경하세요.")
        return
    
    # 카메라 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # 실제 해상도 확인
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("\n=== 웹캠 인식 시작 ===")
    print(f"카메라: {args.camera}")
    print(f"해상도: {actual_width}x{actual_height}")
    print(f"Confidence threshold: {args.confidence}")
    print("종료하려면 'q' 키를 누르세요")
    print("-" * 40)
    
    # 통계 변수
    total_frames = 0
    total_people_detected = 0
    max_people_in_frame = 0
    
    # FPS 계산
    fps_start_time = datetime.now()
    fps_frame_count = 0
    
    # 이미지 저장 디렉토리
    if args.save:
        save_dir = "detected_images"
        os.makedirs(save_dir, exist_ok=True)
        print(f"감지된 이미지는 '{save_dir}' 폴더에 저장됩니다.")
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            total_frames += 1
            
            # YOLO 추론 (person 클래스만, 커스텀 confidence)
            results = model(
                frame,
                verbose=False,
                classes=[0],  # person class only
                conf=args.confidence
            )[0]
            
            # Person 감지 결과 그리기
            person_count = 0
            detections = []
            
            for box in results.boxes:
                if int(box.cls) == 0:  # person class
                    person_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf)
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": confidence
                    })
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
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
            
            # 통계 업데이트
            total_people_detected += person_count
            if person_count > max_people_in_frame:
                max_people_in_frame = person_count
            
            # 인원수 표시
            count_text = f"People: {person_count}"
            cv2.putText(
                frame,
                count_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )
            
            # 평균 인원수 표시
            avg_people = total_people_detected / total_frames if total_frames > 0 else 0
            avg_text = f"Avg: {avg_people:.1f} | Max: {max_people_in_frame}"
            cv2.putText(
                frame,
                avg_text,
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
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
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # 설정 정보 표시
            info_text = f"Conf: {args.confidence} | Model: {os.path.basename(args.model)}"
            cv2.putText(
                frame,
                info_text,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            
            # 이미지 저장 (사람이 감지된 경우)
            if args.save and person_count > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{save_dir}/detected_{timestamp}_{person_count}people.jpg"
                cv2.imwrite(filename, frame)
            
            # 프레임 표시
            cv2.imshow("Person Detection - Webcam (Advanced)", frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n종료합니다...")
                break
                
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    finally:
        # 리소스 해제
        cap.release()
        cv2.destroyAllWindows()
        
        # 통계 출력
        print("\n=== 통계 ===")
        print(f"총 프레임 수: {total_frames}")
        print(f"평균 인원수: {total_people_detected / total_frames if total_frames > 0 else 0:.2f}")
        print(f"최대 인원수: {max_people_in_frame}")
        print("카메라가 종료되었습니다.")

if __name__ == "__main__":
    main()


