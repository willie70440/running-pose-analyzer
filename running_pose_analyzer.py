import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class RunningMetrics:
    """跑步指標數據結構"""
    trunk_lean_angle: float  # 軀幹前傾角度
    knee_lift_angle: float   # 膝蓋抬升角度
    foot_strike_pattern: str # 著地模式
    arm_swing_angle: float   # 手臂擺動角度
    cadence: float          # 步頻 (steps/min)
    stride_length: float    # 步幅相對長度
    overall_score: float    # 綜合評分

class RunningPoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 步數和時間追蹤
        self.step_timestamps = []
        self.foot_positions_history = []
        self.frame_count = 0
        self.fps = 30  # 默認 FPS
        
    def calculate_angle(self, point1: Tuple, point2: Tuple, point3: Tuple) -> float:
        """計算三點間的角度"""
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        return angle
    
    def get_trunk_lean_angle(self, landmarks) -> float:
        """計算軀幹前傾角度"""
        shoulder_mid = (
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + 
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + 
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
        )
        
        hip_mid = (
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x + 
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y + 
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
        )
        
        # 計算軀幹與垂直線的夾角
        trunk_vector = np.array([hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]])
        vertical_vector = np.array([0, 1])
        
        cos_angle = np.dot(trunk_vector, vertical_vector) / np.linalg.norm(trunk_vector)
        angle = math.degrees(math.acos(abs(cos_angle)))
        
        return angle
    
    def analyze_knee_lift(self, landmarks) -> Tuple[float, float]:
        """分析膝蓋抬升角度"""
        left_knee_angle = self.calculate_angle(
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
        )
        
        right_knee_angle = self.calculate_angle(
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
        )
        
        return left_knee_angle, right_knee_angle
    
    def analyze_foot_strike(self, landmarks) -> str:
        """分析腳部著地模式"""
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_heel = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value]
        right_heel = landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value]
        left_foot_index = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        right_foot_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
        
        # 計算腳踝相對於腳尖和腳跟的位置
        left_heel_ankle_diff = left_heel.y - left_ankle.y
        left_toe_ankle_diff = left_foot_index.y - left_ankle.y
        
        right_heel_ankle_diff = right_heel.y - right_ankle.y
        right_toe_ankle_diff = right_foot_index.y - right_ankle.y
        
        # 判斷著地模式
        if abs(left_heel_ankle_diff) < 0.02 and abs(right_heel_ankle_diff) < 0.02:
            return "midfoot"  # 中足著地
        elif left_heel_ankle_diff > 0.02 or right_heel_ankle_diff > 0.02:
            return "heel"     # 腳跟著地
        else:
            return "forefoot"  # 前掌著地
    
    def analyze_arm_swing(self, landmarks) -> float:
        """分析手臂擺動角度"""
        left_arm_angle = self.calculate_angle(
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y)
        )
        
        right_arm_angle = self.calculate_angle(
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
        )
        
        return (left_arm_angle + right_arm_angle) / 2
    
    def calculate_cadence_and_stride(self, landmarks) -> Tuple[float, float]:
        """計算步頻和步幅"""
        # 追蹤腳部位置變化來檢測步數
        left_foot_y = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        right_foot_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        
        current_time = self.frame_count / self.fps
        
        # 檢測腳部觸地（簡化版本）
        if len(self.foot_positions_history) > 5:
            # 檢測左腳觸地
            if (left_foot_y > self.foot_positions_history[-1]['left_foot_y'] and 
                left_foot_y - min([pos['left_foot_y'] for pos in self.foot_positions_history[-5:]]) > 0.05):
                self.step_timestamps.append(current_time)
            
            # 檢測右腳觸地
            if (right_foot_y > self.foot_positions_history[-1]['right_foot_y'] and 
                right_foot_y - min([pos['right_foot_y'] for pos in self.foot_positions_history[-5:]]) > 0.05):
                self.step_timestamps.append(current_time)
        
        self.foot_positions_history.append({
            'left_foot_y': left_foot_y,
            'right_foot_y': right_foot_y,
            'time': current_time
        })
        
        # 計算步頻 (每分鐘步數)
        if len(self.step_timestamps) >= 2:
            time_window = 10  # 秒
            recent_steps = [t for t in self.step_timestamps if current_time - t <= time_window]
            cadence = len(recent_steps) * (60 / time_window)
        else:
            cadence = 0
        
        # 計算相對步幅（基於身體比例）
        hip_to_ankle_distance = abs(
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y - 
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        )
        
        if len(self.foot_positions_history) >= 2:
            foot_displacement = abs(
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x - 
                self.foot_positions_history[-2]['left_foot_y']  # 簡化計算
            )
            relative_stride = foot_displacement / hip_to_ankle_distance if hip_to_ankle_distance > 0 else 0
        else:
            relative_stride = 0
        
        return cadence, relative_stride
    
    def calculate_overall_score(self, metrics: RunningMetrics) -> float:
        """計算綜合評分 (0-100)"""
        score = 100
        
        # 軀幹前傾評分 (理想範圍: 5-15度)
        if metrics.trunk_lean_angle < 5:
            score -= (5 - metrics.trunk_lean_angle) * 2
        elif metrics.trunk_lean_angle > 15:
            score -= (metrics.trunk_lean_angle - 15) * 3
        
        # 膝蓋抬升評分 (理想範圍: 90-120度)
        if metrics.knee_lift_angle < 90:
            score -= (90 - metrics.knee_lift_angle) * 0.5
        elif metrics.knee_lift_angle > 140:
            score -= (metrics.knee_lift_angle - 140) * 0.3
        
        # 著地模式評分
        if metrics.foot_strike_pattern == "heel":
            score -= 15  # 腳跟著地扣分較多
        elif metrics.foot_strike_pattern == "forefoot":
            score -= 5   # 前掌著地輕微扣分
        # midfoot 不扣分
        
        # 手臂擺動評分 (理想範圍: 80-100度)
        if metrics.arm_swing_angle < 70 or metrics.arm_swing_angle > 110:
            score -= 10
        
        # 步頻評分 (理想範圍: 170-190 steps/min)
        if metrics.cadence > 0:
            if metrics.cadence < 160 or metrics.cadence > 200:
                score -= 15
        
        return max(0, score)
    
    def analyze_frame(self, frame) -> Tuple[Optional[RunningMetrics], Any]:
        """分析單一幀的跑步姿勢，返回指標和pose landmarks"""
        self.frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None, None

        landmarks = results.pose_landmarks.landmark

        # 計算各項指標
        trunk_lean = self.get_trunk_lean_angle(landmarks)
        left_knee, right_knee = self.analyze_knee_lift(landmarks)
        knee_lift = (left_knee + right_knee) / 2
        foot_strike = self.analyze_foot_strike(landmarks)
        arm_swing = self.analyze_arm_swing(landmarks)
        cadence, stride = self.calculate_cadence_and_stride(landmarks)

        # 創建指標對象
        metrics = RunningMetrics(
            trunk_lean_angle=trunk_lean,
            knee_lift_angle=knee_lift,
            foot_strike_pattern=foot_strike,
            arm_swing_angle=arm_swing,
            cadence=cadence,
            stride_length=stride,
            overall_score=0  # 暫時設為0
        )

        # 計算綜合評分
        metrics.overall_score = self.calculate_overall_score(metrics)

        return metrics, results.pose_landmarks
    
    def draw_pose_analysis(self, frame, metrics: RunningMetrics, pose_landmarks=None):
        """在畫面上繪製姿勢分析結果和33個關節點"""
        height, width = frame.shape[:2]

        # 繪製 33 個關節點和連接線
        if pose_landmarks:
            # 使用 MediaPipe 的繪製工具顯示所有33個關節點
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0),  # 綠色關節點
                    thickness=3,
                    circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 255),  # 粉紫色連接線
                    thickness=2
                )
            )

            # 特別標示重要關節點
            landmarks = pose_landmarks.landmark

            # 標示肩膀中心點 (用於軀幹分析)
            shoulder_mid_x = int((landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x +
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2 * width)
            shoulder_mid_y = int((landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2 * height)
            cv2.circle(frame, (shoulder_mid_x, shoulder_mid_y), 6, (0, 0, 255), -1)  # 紅色圓點
            cv2.putText(frame, "Shoulder Center", (shoulder_mid_x - 30, shoulder_mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # 標示髖部中心點
            hip_mid_x = int((landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x +
                           landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2 * width)
            hip_mid_y = int((landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y +
                           landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2 * height)
            cv2.circle(frame, (hip_mid_x, hip_mid_y), 6, (0, 0, 255), -1)
            cv2.putText(frame, "Hip Center", (hip_mid_x - 25, hip_mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # 繪製軀幹角度線
            cv2.line(frame, (shoulder_mid_x, shoulder_mid_y), (hip_mid_x, hip_mid_y), (255, 255, 0), 3)

            # 標示腳部關節點 (用於著地分析)
            for landmark_name in ['LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']:
                landmark = landmarks[getattr(self.mp_pose.PoseLandmark, landmark_name).value]
                x, y = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(frame, (x, y), 4, (255, 165, 0), -1)  # 橙色圓點

        # 在畫面右上角顯示關節點總數
        cv2.putText(frame, "33 Pose Landmarks Detected",
                   (width - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 繪製分析結果文字 (移到左下角)
        y_offset = height - 150
        cv2.putText(frame, f"Overall Score: {metrics.overall_score:.1f}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        y_offset += 30
        cv2.putText(frame, f"Trunk Lean: {metrics.trunk_lean_angle:.1f}°",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_offset += 25
        cv2.putText(frame, f"Knee Lift: {metrics.knee_lift_angle:.1f}°",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_offset += 25
        cv2.putText(frame, f"Foot Strike: {metrics.foot_strike_pattern}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_offset += 25
        cv2.putText(frame, f"Cadence: {metrics.cadence:.0f} spm",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame
    
    def get_recommendations(self, metrics: RunningMetrics) -> List[str]:
        """根據分析結果生成建議"""
        recommendations = []
        
        if metrics.trunk_lean_angle < 5:
            recommendations.append("軀幹稍微向前傾斜，有助於重心前移和推進效率")
        elif metrics.trunk_lean_angle > 15:
            recommendations.append("軀幹過度前傾，容易造成腰部負擔，建議保持挺胸")
        
        if metrics.knee_lift_angle < 90:
            recommendations.append("膝蓋抬升不足，增加大腿前側肌群訓練")
        elif metrics.knee_lift_angle > 140:
            recommendations.append("膝蓋抬升過高，浪費體力，調整為中等抬腿高度")
        
        if metrics.foot_strike_pattern == "heel":
            recommendations.append("建議改為中足著地，減少膝蓋衝擊")
        
        if metrics.arm_swing_angle < 70:
            recommendations.append("手臂擺動幅度過小，增加推進力")
        elif metrics.arm_swing_angle > 110:
            recommendations.append("手臂擺動過大，浪費能量")
        
        if metrics.cadence > 0 and metrics.cadence < 160:
            recommendations.append("步頻偏低，嘗試增加到 170-180 步/分鐘")
        elif metrics.cadence > 200:
            recommendations.append("步頻過高，可能導致過度疲勞")
        
        return recommendations

    def analyze_video_with_visualization(self, video_path: str, output_path: str = None, show_live: bool = True):
        """分析影片並顯示33個關節點的可視化結果

        Args:
            video_path: 原始影片路徑（不會被修改）
            output_path: 可選的輸出路徑，如果提供則儲存帶關節點的副本
            show_live: 是否即時顯示分析過程
        """
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"無法開啟影片檔案: {video_path}")
                return None, []

            # 取得影片資訊
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"📹 原始影片: {video_path} ({width}x{height}, {fps}fps)")
            print(f"🔍 分析模式: {'即時顯示' if show_live else '背景處理'}")
            if output_path:
                print(f"💾 將儲存標記版本至: {output_path}")
            else:
                print("❗ 只分析不儲存，原始影片不會被修改")

            # 只在指定輸出路徑時才建立 VideoWriter
            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            all_metrics = []
            frame_count = 0

            print("開始分析影片，按 'q' 鍵退出...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                try:
                    # 分析當前幀
                    metrics, pose_landmarks = self.analyze_frame(frame)

                    if metrics and pose_landmarks:
                        all_metrics.append(metrics)

                        # 繪製33個關節點和分析結果
                        annotated_frame = self.draw_pose_analysis(frame, metrics, pose_landmarks)

                        # 顯示幀數資訊
                        cv2.putText(annotated_frame, f"Frame: {frame_count}",
                                   (width - 150, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        annotated_frame = frame
                        cv2.putText(annotated_frame, "No pose detected",
                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # 保存到輸出檔案
                    if output_path:
                        out.write(annotated_frame)

                    # 即時顯示
                    if show_live:
                        cv2.imshow('Running Pose Analysis with 33 Landmarks', annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                except Exception as e:
                    print(f"處理第 {frame_count} 幀時發生錯誤: {e}")
                    continue

            cap.release()
            if output_path:
                out.release()
            if show_live:
                cv2.destroyAllWindows()

            # 計算平均指標
            if all_metrics:
                valid_cadences = [m.cadence for m in all_metrics if m.cadence > 0 and m.cadence < 300]

                avg_metrics = RunningMetrics(
                    trunk_lean_angle=np.mean([m.trunk_lean_angle for m in all_metrics]),
                    knee_lift_angle=np.mean([m.knee_lift_angle for m in all_metrics]),
                    foot_strike_pattern=max(set([m.foot_strike_pattern for m in all_metrics]),
                                          key=[m.foot_strike_pattern for m in all_metrics].count),
                    arm_swing_angle=np.mean([m.arm_swing_angle for m in all_metrics]),
                    cadence=np.mean(valid_cadences) if valid_cadences else 0,
                    stride_length=np.mean([m.stride_length for m in all_metrics]),
                    overall_score=np.mean([m.overall_score for m in all_metrics])
                )

                recommendations = self.get_recommendations(avg_metrics)

                print(f"\n=== 分析完成 ===")
                print(f"處理了 {frame_count} 幀")
                print(f"檢測到姿勢的幀數: {len(all_metrics)}")
                print(f"檢測成功率: {len(all_metrics)/frame_count*100:.1f}%")
                print(f"\n綜合評分: {avg_metrics.overall_score:.1f}/100")

                if output_path:
                    print(f"已儲存分析影片至: {output_path}")

                return avg_metrics, recommendations

            return None, []

        except Exception as e:
            print(f"影片分析過程中發生錯誤: {e}")
            return None, []

# 使用範例
def analyze_running_video(video_path: str):
    """分析跑步影片的完整流程"""
    try:
        analyzer = RunningPoseAnalyzer()
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"無法開啟影片檔案: {video_path}")
            return None, []

        all_metrics = []
        frame_count = 0
        max_frames = 300  # 限制最大幀數，避免處理過長影片

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 每隔幾幀處理一次，提高處理速度
            if frame_count % 3 == 0:
                try:
                    metrics, pose_landmarks = analyzer.analyze_frame(frame)
                    if metrics:
                        all_metrics.append(metrics)
                except Exception as e:
                    print(f"分析第 {frame_count} 幀時發生錯誤: {e}")
                    continue

        cap.release()

        # 計算平均指標
        if all_metrics:
            # 過濾掉異常值
            valid_cadences = [m.cadence for m in all_metrics if m.cadence > 0 and m.cadence < 300]

            avg_metrics = RunningMetrics(
                trunk_lean_angle=np.mean([m.trunk_lean_angle for m in all_metrics]),
                knee_lift_angle=np.mean([m.knee_lift_angle for m in all_metrics]),
                foot_strike_pattern=max(set([m.foot_strike_pattern for m in all_metrics]),
                                      key=[m.foot_strike_pattern for m in all_metrics].count),
                arm_swing_angle=np.mean([m.arm_swing_angle for m in all_metrics]),
                cadence=np.mean(valid_cadences) if valid_cadences else 0,
                stride_length=np.mean([m.stride_length for m in all_metrics]),
                overall_score=np.mean([m.overall_score for m in all_metrics])
            )

            recommendations = analyzer.get_recommendations(avg_metrics)
            return avg_metrics, recommendations

        return None, []

    except Exception as e:
        print(f"影片分析過程中發生錯誤: {e}")
        return None, []