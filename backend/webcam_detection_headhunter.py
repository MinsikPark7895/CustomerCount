"""
HeadHunter 모델을 사용한 실시간 머리 인식
CVPR 2021 "Tracking Pedestrian Heads in Dense Crowd" 기반
탑뷰(항공 샷) 환경에 최적화된 Head Detection

사용법:
    python webcam_detection_headhunter.py

주의:
    HeadHunter 모델 설치가 필요할 수 있습니다.
    pip install head_detection
    또는 GitHub에서 직접 설치: https://github.com/Sentient07/HeadHunter
"""

import cv2
import os
from datetime import datetime
import time
import numpy as np

# HeadHunter 모델 import 시도
HEADHUNTER_AVAILABLE = False
HEADHUNTER_ERROR = None

try:
    # 먼저 필요한 기본 패키지 확인
    try:
        import torch
        import torchvision
    except ImportError as e:
        raise ImportError(f"필수 패키지가 설치되지 않았습니다: {e}\n다음 명령어로 설치하세요: pip install torch torchvision")
    
    # albumentations 확인
    try:
        import albumentations
    except ImportError:
        raise ImportError("albumentations 패키지가 설치되지 않았습니다.\n다음 명령어로 설치하세요: pip install albumentations")
    
    # HeadHunter의 실제 구조 확인
    import sys
    headhunter_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "HeadHunter-master")
    if os.path.exists(headhunter_path):
        sys.path.insert(0, headhunter_path)
        print(f"HeadHunter 경로 추가: {headhunter_path}")
    
    # HeadHunter 패키지가 설치되어 있는지 확인
    try:
        import head_detection
        print("head_detection 패키지를 찾았습니다.")
    except ImportError:
        # HeadHunter-master 디렉토리에서 직접 설치 시도
        if os.path.exists(headhunter_path) and os.path.exists(os.path.join(headhunter_path, "setup.py")):
            print(f"\n경고: head_detection 패키지가 설치되지 않았습니다.")
            print(f"HeadHunter-master 디렉토리에서 다음 명령어를 실행하세요:")
            print(f"  cd {headhunter_path}")
            print(f"  pip install .")
            print(f"\n또는 필요한 패키지를 설치하세요:")
            print(f"  pip install albumentations torch torchvision")
            raise ImportError("head_detection 패키지가 설치되지 않았습니다.")
        else:
            raise ImportError("HeadHunter-master 디렉토리를 찾을 수 없습니다.")
    
    # 먼저 head_detection.utils만 import 시도 (가장 기본적인 모듈)
    try:
        from head_detection.utils import get_state_dict, to_torch
        print("head_detection.utils 모듈을 성공적으로 import했습니다.")
    except (ImportError, AttributeError) as e:
        error_msg = str(e)
        if 'albumentations' in error_msg.lower():
            raise ImportError(f"albumentations 패키지가 필요합니다: {e}\n다음 명령어로 설치하세요: pip install albumentations")
        elif 'imread' in error_msg or 'scipy.misc' in error_msg:
            print(f"경고: scipy.misc 호환성 문제: {e}")
            print("HeadHunter의 scipy.misc 의존성 문제로 인해 import 실패")
            raise ImportError(f"scipy.misc 호환성 문제: {e}")
        else:
            raise ImportError(f"head_detection.utils import 실패: {e}")
    
    # models와 data 모듈은 존재하지 않을 수 있으므로 확인
    try:
        from head_detection.models.head_detect import customRCNN
        from head_detection.data import cfg_res50_4fpn, combined_anchors
        HEADHUNTER_AVAILABLE = True
        print("✓ HeadHunter 모듈을 성공적으로 로드했습니다!")
    except (ImportError, ModuleNotFoundError) as e:
        # models나 data 모듈이 없으면 HeadHunter 사용 불가
        HEADHUNTER_AVAILABLE = False
        HEADHUNTER_ERROR = str(e)
        print(f"\n경고: HeadHunter의 models/data 모듈을 찾을 수 없습니다: {e}")
        print("HeadHunter 저장소가 불완전한 것 같습니다.")
        print("models/ 및 data/ 디렉토리가 필요합니다.")
        print("\n다음 중 하나를 시도하세요:")
        print("1. HeadHunter 저장소를 완전히 다운로드 (models/, data/ 디렉토리 포함)")
        print("2. HeadHunter-master 디렉토리에서 'pip install .' 실행")
        print("3. GitHub에서 완전한 HeadHunter 저장소 클론")
        raise ImportError(f"HeadHunter 모듈 불완전: {e}")
        
except (ImportError, AttributeError, ModuleNotFoundError) as e:
    HEADHUNTER_AVAILABLE = False
    HEADHUNTER_ERROR = str(e)
    print("\n" + "="*60)
    print("경고: HeadHunter 모듈을 사용할 수 없습니다.")
    print("="*60)
    print(f"오류: {e}")
    print("\nHeadHunter를 사용하려면 다음 단계를 따르세요:")
    print("1. 필요한 패키지 설치:")
    print("   pip install albumentations torch torchvision")
    print("2. HeadHunter-master 디렉토리에서 패키지 설치:")
    print(f"   cd {headhunter_path if 'headhunter_path' in locals() else 'HeadHunter-master'}")
    print("   pip install .")
    print("3. HeadHunter 저장소가 완전한지 확인 (models/, data/ 디렉토리 필요)")
    print("4. 모델 가중치 파일이 있는지 확인")
    print("\n" + "="*60)
    print("대안으로 YOLOv8 기반 Head Detection을 사용합니다...")
    print("="*60 + "\n")
    
    # 대안: YOLOv8 사용
    try:
        from ultralytics import YOLO
        import torch
        
        # PyTorch 2.6+ 호환성
        _original_torch_load = torch.load
        def _patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)
        torch.load = _patched_torch_load
    except ImportError as yolo_error:
        print(f"YOLOv8도 사용할 수 없습니다: {yolo_error}")
        print("프로그램을 종료합니다.")
        import sys
        sys.exit(1)

def main():
    """메인 함수: 웹캠에서 실시간 머리 인식"""
    
    # HeadHunter 모델 초기화
    detector = None
    use_headhunter = False
    
    if HEADHUNTER_AVAILABLE:
        print("\n" + "="*60)
        print("HeadHunter 모델 로딩 중...")
        print("="*60)
        try:
            # HeadHunter 모델 생성 (test.py 참고)
            cfg = cfg_res50_4fpn
            combined_cfg = {**cfg, **combined_anchors}
            
            # 모델 가중치 경로 확인
            model_weight_path = os.getenv("HEADHUNTER_WEIGHTS", None)
            if model_weight_path is None:
                # 기본 경로 시도
                headhunter_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "HeadHunter-master")
                possible_paths = [
                    os.path.join(headhunter_dir, "weights", "headhunter.pth"),
                    os.path.join(headhunter_dir, "weights", "model.pth"),
                    os.path.join(headhunter_dir, "weights", "headhunter_combined.pth"),
                    os.path.join(headhunter_dir, "weights", "res50_4fpn.pth"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        model_weight_path = path
                        print(f"모델 가중치 파일을 찾았습니다: {path}")
                        break
            
            if model_weight_path is None or not os.path.exists(model_weight_path):
                print("\n" + "!"*60)
                print("경고: HeadHunter 모델 가중치를 찾을 수 없습니다.")
                print("!"*60)
                print("다음 경로에서 가중치 파일을 찾았습니다:")
                headhunter_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "HeadHunter-master")
                weights_dir = os.path.join(headhunter_dir, "weights")
                if os.path.exists(weights_dir):
                    print(f"  {weights_dir}")
                    print("  이 디렉토리의 파일들:")
                    for f in os.listdir(weights_dir):
                        print(f"    - {f}")
                print("\n다음 중 하나를 시도하세요:")
                print("1. 환경 변수 HEADHUNTER_WEIGHTS를 설정하세요")
                print("2. HeadHunter 논문 페이지에서 사전 학습된 가중치를 다운로드하세요")
                print("3. 가중치 파일을 weights/ 디렉토리에 저장하세요")
                print("\nHeadHunter를 사용할 수 없으므로 YOLOv8 기반 대안을 사용합니다...")
                use_headhunter = False
            else:
                # 모델 생성
                kwargs = {
                    'min_size': 800,
                    'max_size': 1400,
                    'box_score_thresh': 0.3,
                    'box_nms_thresh': 0.5,
                    'box_detections_per_img': 300
                }
                detector = customRCNN(cfg=combined_cfg, **kwargs)
                
                # 가중치 로드
                if torch.cuda.is_available():
                    detector = detector.cuda()
                    new_state_dict = get_state_dict(detector, model_weight_path)
                    detector.load_state_dict(new_state_dict, strict=True)
                    print("CUDA를 사용하여 모델을 로드했습니다.")
                else:
                    print("경고: CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다 (매우 느림).")
                    new_state_dict = get_state_dict(detector, model_weight_path)
                    detector.load_state_dict(new_state_dict, strict=True)
                
                detector = detector.eval()
                print(f"✓ HeadHunter 모델 로드 완료! (가중치: {model_weight_path})")
                use_headhunter = True
        except Exception as e:
            print(f"\n오류: HeadHunter 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            print("\nYOLOv8 기반 대안을 사용합니다...")
            use_headhunter = False
    else:
        print("\n" + "="*60)
        print("HeadHunter를 사용할 수 없습니다.")
        print("="*60)
        if HEADHUNTER_ERROR:
            print(f"오류: {HEADHUNTER_ERROR}")
        print("\nHeadHunter를 사용하려면 위의 설치 단계를 따르세요.")
        print("현재는 YOLOv8 기반 대안을 사용합니다.")
        print("="*60 + "\n")
    
    # YOLOv8 대안 모델 초기화 (HeadHunter가 없을 경우)
    if not use_headhunter:
        print("\nYOLOv8 모델 로딩 중...")
        model_path = os.getenv("MODEL_PATH", "yolov8n.pt")
        
        # 모델 경로 확인
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_full_path = os.path.join(script_dir, model_path)
            if os.path.exists(model_full_path):
                model_path = model_full_path
            elif os.path.exists(model_path):
                model_path = os.path.abspath(model_path)
        
        try:
            model = YOLO(model_path)
            print("YOLOv8 모델 로드 완료!")
            print("주의: YOLOv8은 Person Detection 모델입니다.")
            print("Head Detection 전용 모델로 Fine-tuning하는 것을 권장합니다.")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return
    
    # 웹캠 초기화
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
    
    # 카메라 해상도 설정
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
    print("\n=== Head Detection 시작 ===")
    print("종료하려면 'q' 키를 누르세요")
    print("-" * 40)
    
    # FPS 계산을 위한 변수
    fps_start_time = datetime.now()
    fps_frame_count = 0
    
    # OpenCV 창 생성
    window_name = "Head Detection - Webcam (HeadHunter)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_width, actual_height)
    
    # Confidence threshold (HeadHunter가 없을 경우 YOLO용)
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.20"))  # Head Detection은 낮은 threshold 권장
    
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
            
            # Head Detection 수행
            head_count = 0
            detections = []
            
            if use_headhunter and detector is not None:
                # HeadHunter 모델 사용 (test.py 참고)
                try:
                    # 이미지를 torch tensor로 변환
                    image_tensor = to_torch(frame)
                    
                    # 추론 수행
                    with torch.no_grad():
                        outputs = detector(image_tensor)
                    
                    # 결과 파싱 (test.py 참고)
                    cpu_device = torch.device("cpu")
                    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                    
                    if len(outputs) > 0:
                        output = outputs[0]
                        boxes = output['boxes'].cpu().numpy()
                        scores = output['scores'].cpu().numpy()
                        
                        # Confidence threshold 적용
                        conf_thresh = float(os.getenv("HEADHUNTER_CONF_THRESH", "0.3"))
                        valid_indices = scores >= conf_thresh
                        boxes = boxes[valid_indices]
                        scores = scores[valid_indices]
                        
                        # 바운딩 박스 그리기
                        for box, score in zip(boxes, scores):
                            x1, y1, x2, y2 = box.astype(int)
                            head_count += 1
                            
                            # 바운딩 박스 그리기
                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),  # 초록색
                                2
                            )
                            
                            # 라벨 표시
                            label = f"Head {score:.2f}"
                            cv2.putText(
                                frame,
                                label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )
                except Exception as e:
                    print(f"HeadHunter 추론 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                # YOLOv8 대안 사용 (Person Detection)
                try:
                    results = model(frame, verbose=False, classes=[0], conf=confidence_threshold)[0]  # class 0 = person
                    
                    for box in results.boxes:
                        if int(box.cls) == 0:  # person class
                            head_count += 1
                            
                            # 바운딩 박스 좌표
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = float(box.conf)
                            
                            # Person Detection이므로 상단 부분만 강조 (머리 영역 추정)
                            # 전체 박스의 상단 20%를 머리로 간주
                            head_height = int((y2 - y1) * 0.2)
                            head_y1 = int(y1)
                            head_y2 = int(y1 + head_height)
                            
                            # 전체 사람 박스 (연한 색)
                            cv2.rectangle(
                                frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (100, 100, 100),  # 회색
                                1
                            )
                            
                            # 머리 영역 강조 (진한 색)
                            cv2.rectangle(
                                frame,
                                (int(x1), head_y1),
                                (int(x2), head_y2),
                                (0, 255, 0),  # 초록색
                                2
                            )
                            
                            # 라벨 표시
                            label = f"Head {confidence:.2f}"
                            cv2.putText(
                                frame,
                                label,
                                (int(x1), head_y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )
                except Exception as e:
                    print(f"YOLO 추론 오류: {e}")
                    continue
            
            # 인원수 표시 (화면 상단)
            count_text = f"Head Count: {head_count}"
            cv2.putText(
                frame,
                count_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )
            
            # 모델 정보 표시
            model_text = "HeadHunter" if use_headhunter else f"YOLOv8 (Person->Head 추정)"
            cv2.putText(
                frame,
                model_text,
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
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
            
            # 안내 메시지
            if not use_headhunter:
                info_text = "Note: Using Person Detection (Head Detection model recommended)"
                cv2.putText(
                    frame,
                    info_text,
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
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
