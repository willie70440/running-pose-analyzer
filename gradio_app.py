#!/usr/bin/env python3
"""
Gradio 版跑步姿勢分析應用
使用現有的 running_pose_analyzer.py 進行後端分析
"""

import gradio as gr
import os
import tempfile
import shutil
import json
import csv
from datetime import datetime
from running_pose_analyzer import RunningPoseAnalyzer
import logging

# 設定 logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioRunningAnalyzer:
    def __init__(self):
        self.analyzer = None
        
    def get_analyzer(self):
        """延遲初始化分析器"""
        if self.analyzer is None:
            logger.info("初始化 RunningPoseAnalyzer...")
            self.analyzer = RunningPoseAnalyzer()
        return self.analyzer
    
    def analyze_video_with_detailed_data(self, analyzer, video_path, output_path):
        """進行詳細分析並收集所有幀數據"""
        import cv2
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"無法開啟影片檔案: {video_path}")
                return None
            
            # 取得影片資訊
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"影片資訊: {width}x{height}, {fps}fps, {total_frames} 幀")
            
            # 建立輸出影片
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 收集所有幀數據
            all_frame_data = []
            all_metrics = []
            frame_count = 0
            
            # MediaPipe 關節點名稱對應
            landmark_names = [
                'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
                'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
                'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
                'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
                'right_heel', 'left_foot_index', 'right_foot_index'
            ]
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = frame_count / fps
                
                # 分析當前幀
                metrics, pose_landmarks = analyzer.analyze_frame(frame)
                
                frame_data = {
                    'frame_number': frame_count,
                    'timestamp': current_time,
                    'landmarks': [],
                    'metrics': None
                }
                
                if pose_landmarks and metrics:
                    # 收集33個關節點數據
                    landmarks_data = []
                    for i, landmark in enumerate(pose_landmarks.landmark):
                        landmark_data = {
                            'id': i,
                            'name': landmark_names[i] if i < len(landmark_names) else f'landmark_{i}',
                            'x': float(landmark.x),
                            'y': float(landmark.y),
                            'z': float(landmark.z),
                            'visibility': float(landmark.visibility)
                        }
                        landmarks_data.append(landmark_data)
                    
                    frame_data['landmarks'] = landmarks_data
                    frame_data['metrics'] = {
                        'trunk_lean_angle': float(metrics.trunk_lean_angle),
                        'knee_lift_angle': float(metrics.knee_lift_angle),
                        'foot_strike_pattern': metrics.foot_strike_pattern,
                        'arm_swing_angle': float(metrics.arm_swing_angle),
                        'cadence': float(metrics.cadence),
                        'stride_length': float(metrics.stride_length),
                        'overall_score': float(metrics.overall_score)
                    }
                    
                    all_metrics.append(metrics)
                    
                    # 繪製帶關節點的幀
                    annotated_frame = analyzer.draw_pose_analysis(frame, metrics, pose_landmarks)
                    out.write(annotated_frame)
                else:
                    # 沒有檢測到姿勢的幀
                    out.write(frame)
                
                all_frame_data.append(frame_data)
            
            cap.release()
            out.release()
            
            # 計算摘要指標
            if all_metrics:
                import numpy as np
                
                valid_cadences = [m.cadence for m in all_metrics if m.cadence > 0 and m.cadence < 300]
                
                from running_pose_analyzer import RunningMetrics
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
                
                return {
                    'video_info': {
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'total_frames': total_frames,
                        'duration': total_frames / fps,
                        'detected_frames': len(all_metrics)
                    },
                    'frames': all_frame_data,
                    'summary_metrics': avg_metrics,
                    'recommendations': recommendations
                }
            
            return None
            
        except Exception as e:
            logger.error(f"詳細分析過程發生錯誤: {e}")
            return None
    
    def export_data_files(self, detailed_data):
        """匯出 JSON 和 CSV 檔案"""
        try:
            # 匯出 JSON 檔案（固定名稱）
            json_filename = "latest_pose_data.json"
            json_data = {
                'video_info': detailed_data['video_info'],
                'frames': detailed_data['frames'],
                'summary': {
                    'trunk_lean_angle': float(detailed_data['summary_metrics'].trunk_lean_angle),
                    'knee_lift_angle': float(detailed_data['summary_metrics'].knee_lift_angle),
                    'foot_strike_pattern': detailed_data['summary_metrics'].foot_strike_pattern,
                    'arm_swing_angle': float(detailed_data['summary_metrics'].arm_swing_angle),
                    'cadence': float(detailed_data['summary_metrics'].cadence),
                    'stride_length': float(detailed_data['summary_metrics'].stride_length),
                    'overall_score': float(detailed_data['summary_metrics'].overall_score)
                },
                'recommendations': detailed_data['recommendations']
            }
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            # 匯出 CSV 檔案（固定名稱）
            csv_filename = "latest_pose_data.csv"
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # CSV 標題行
                header = ['frame', 'timestamp']
                # 添加33個關節點的 x, y, z, visibility
                landmark_names = [
                    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
                    'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
                    'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
                    'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
                    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
                    'right_heel', 'left_foot_index', 'right_foot_index'
                ]
                
                for name in landmark_names:
                    header.extend([f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_visibility'])
                
                # 添加跑步指標
                header.extend(['trunk_lean_angle', 'knee_lift_angle', 'foot_strike_pattern', 
                              'arm_swing_angle', 'cadence', 'stride_length', 'overall_score'])
                
                writer.writerow(header)
                
                # 寫入數據行
                for frame_data in detailed_data['frames']:
                    row = [frame_data['frame_number'], frame_data['timestamp']]
                    
                    # 添加關節點數據
                    if frame_data['landmarks']:
                        for landmark in frame_data['landmarks']:
                            row.extend([landmark['x'], landmark['y'], landmark['z'], landmark['visibility']])
                    else:
                        # 沒有檢測到關節點時填入空值
                        row.extend([''] * (33 * 4))
                    
                    # 添加跑步指標
                    if frame_data['metrics']:
                        metrics = frame_data['metrics']
                        row.extend([
                            metrics['trunk_lean_angle'],
                            metrics['knee_lift_angle'],
                            metrics['foot_strike_pattern'],
                            metrics['arm_swing_angle'],
                            metrics['cadence'],
                            metrics['stride_length'],
                            metrics['overall_score']
                        ])
                    else:
                        row.extend([''] * 7)
                    
                    writer.writerow(row)
            
            return json_filename, csv_filename
            
        except Exception as e:
            logger.error(f"匯出數據檔案失敗: {e}")
            return None, None
    
    def prepare_detailed_analysis_table(self, detailed_data):
        """準備詳細分析數據表格"""
        try:
            table_data = []
            
            # 只顯示有檢測到姿勢的幀（前50幀作為範例）
            valid_frames = [frame for frame in detailed_data['frames'] 
                          if frame['metrics'] is not None]
            
            # 限制顯示數量避免界面過載
            display_frames = valid_frames[:50] if len(valid_frames) > 50 else valid_frames
            
            for frame in display_frames:
                metrics = frame['metrics']
                row = [
                    frame['frame_number'],
                    round(frame['timestamp'], 2),
                    round(metrics['trunk_lean_angle'], 1),
                    round(metrics['knee_lift_angle'], 1),  # 平均膝蓋角度
                    round(metrics['knee_lift_angle'], 1),  # 暫時用同一值
                    round(metrics['arm_swing_angle'], 1),
                    metrics['foot_strike_pattern'],
                    round(metrics['cadence'], 0) if metrics['cadence'] > 0 else 0,
                    round(metrics['overall_score'], 1)
                ]
                table_data.append(row)
            
            return table_data
            
        except Exception as e:
            logger.error(f"準備詳細分析表格失敗: {e}")
            return []
    
    def analyze_video(self, video_file):
        """分析上傳的影片"""
        if video_file is None:
            return None, None, None, "❌ 請上傳影片檔案", "請先選擇要分析的跑步影片", []
        
        try:
            # 使用固定檔案名稱（會自動覆蓋舊檔案）
            output_filename = "latest_analysis.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            
            logger.info(f"開始分析影片: {video_file}")
            logger.info(f"輸出路徑: {output_path}")
            
            # 取得分析器
            analyzer = self.get_analyzer()
            
            # 進行詳細分析並收集所有幀數據
            detailed_data = self.analyze_video_with_detailed_data(analyzer, video_file, output_path)
            
            if not detailed_data or not detailed_data['summary_metrics']:
                return None, None, None, "❌ 分析失敗", "無法在影片中檢測到人體姿勢，請確保影片清晰且人物完整入鏡", []
            
            # 提取摘要數據
            metrics = detailed_data['summary_metrics']
            recommendations = detailed_data['recommendations']
            
            # 格式化分析結果
            results_text = f"""🏃‍♂️ **跑步姿勢分析結果**

📊 **綜合評分**: {metrics.overall_score:.1f}/100

📋 **詳細指標**:
• 軀幹前傾角度: {metrics.trunk_lean_angle:.1f}° {'✅' if 5 <= metrics.trunk_lean_angle <= 15 else '⚠️'}
• 膝蓋抬升角度: {metrics.knee_lift_angle:.1f}° {'✅' if 90 <= metrics.knee_lift_angle <= 140 else '⚠️'}
• 腳部著地模式: {metrics.foot_strike_pattern} {'✅' if metrics.foot_strike_pattern == 'midfoot' else '⚠️'}
• 手臂擺動角度: {metrics.arm_swing_angle:.1f}° {'✅' if 70 <= metrics.arm_swing_angle <= 110 else '⚠️'}
• 步頻: {metrics.cadence:.0f} spm {'✅' if 170 <= metrics.cadence <= 190 else '⚠️' if metrics.cadence > 0 else 'N/A'}
• 相對步幅: {metrics.stride_length:.3f}

📈 **評分標準**:
✅ = 理想範圍  ⚠️ = 需要改善
"""
            
            # 格式化建議
            if recommendations:
                suggestions_text = "💡 **改善建議**:\n\n" + "\n".join([f"• {rec}" for rec in recommendations])
            else:
                suggestions_text = "🎉 **恭喜！您的跑步姿勢很棒！** \n\n繼續保持當前的跑步技術，定期檢測以維持最佳狀態。"
            
            # 匯出數據檔案
            json_file, csv_file = self.export_data_files(detailed_data)
            
            # 檢查輸出檔案是否存在
            if os.path.exists(output_path):
                logger.info(f"分析完成，輸出檔案: {output_path}")
                
                # 複製到當前目錄方便下載（固定名稱）
                local_output = "latest_result.mp4"
                try:
                    shutil.copy2(output_path, local_output)
                    file_size = os.path.getsize(local_output) / (1024*1024)  # MB
                    
                    # 添加檔案資訊到結果
                    results_text += f"\n\n📁 **輸出檔案**:\n"
                    results_text += f"• 影片: `{local_output}` ({file_size:.1f} MB)\n"
                    
                    if json_file:
                        json_size = os.path.getsize(json_file) / 1024  # KB
                        results_text += f"• JSON 數據: `{json_file}` ({json_size:.1f} KB)\n"
                    
                    if csv_file:
                        csv_size = os.path.getsize(csv_file) / 1024  # KB
                        results_text += f"• CSV 數據: `{csv_file}` ({csv_size:.1f} KB)\n"
                    
                    results_text += f"\n📊 **數據統計**:\n"
                    results_text += f"• 總幀數: {detailed_data['video_info']['total_frames']}\n"
                    results_text += f"• 檢測成功幀數: {detailed_data['video_info']['detected_frames']}\n"
                    results_text += f"• 檢測成功率: {detailed_data['video_info']['detected_frames']/detailed_data['video_info']['total_frames']*100:.1f}%"
                    
                    # 準備詳細分析表格
                    detailed_table = self.prepare_detailed_analysis_table(detailed_data)
                    
                    # 回傳檔案路徑和表格數據
                    return local_output, json_file, csv_file, results_text, suggestions_text, detailed_table
                    
                except Exception as copy_error:
                    logger.warning(f"複製檔案失敗: {copy_error}")
                    results_text += f"\n\n📁 **輸出檔案**: `{output_path}`"
                    # 準備詳細分析表格
                    detailed_table = self.prepare_detailed_analysis_table(detailed_data)
                    return output_path, json_file, csv_file, results_text, suggestions_text, detailed_table
            else:
                logger.error("輸出檔案不存在")
                return None, None, None, "❌ 影片生成失敗", "分析完成但無法生成輸出影片", []
                
        except Exception as e:
            logger.error(f"分析過程發生錯誤: {e}")
            error_msg = f"❌ 分析失敗: {str(e)}"
            
            # 根據錯誤類型提供具體建議
            if "OpenCV" in str(e):
                error_suggestion = "影片格式可能不受支援，請嘗試使用 MP4 格式"
            elif "pose_landmarks" in str(e).lower():
                error_suggestion = "無法檢測到人體姿勢，請確保拍攝時人物清晰可見"
            elif "memory" in str(e).lower():
                error_suggestion = "影片檔案過大，請使用較小的影片檔案"
            else:
                error_suggestion = "請檢查影片格式是否正確，建議使用 MP4、MOV 或 AVI 格式"
            
            return None, None, None, error_msg, error_suggestion, []

# 創建分析器實例
gradio_analyzer = GradioRunningAnalyzer()

# 創建 Gradio 介面
def create_interface():
    """創建 Gradio 使用者介面"""
    
    # 自定義 CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-video {
        border-radius: 10px;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    """
    
    # 建立介面
    with gr.Blocks(css=custom_css, title="跑步姿勢分析 AI", theme=gr.themes.Soft()) as interface:
        
        # 標題區域
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 20px;">
            <h1>🏃‍♂️ 跑步姿勢分析 AI</h1>
            <p style="font-size: 18px; margin: 10px 0;">上傳您的跑步影片，獲得專業的姿勢分析和改善建議</p>
            <p style="font-size: 14px; opacity: 0.9;">✨ 33個關節點檢測 | 📊 專業指標分析 | 💡 個人化建議</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 輸入區域
                gr.HTML("<h3>📤 上傳影片</h3>")
                video_input = gr.Video(
                    label="選擇跑步影片",
                    format="mp4"
                )
                
                # 使用說明
                gr.HTML("""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 15px 0;">
                    <h4>📝 拍攝建議：</h4>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>📹 側面拍攝效果最佳</li>
                        <li>🏃‍♂️ 確保全身都在畫面中</li>
                        <li>⏱️ 影片長度建議 10-30 秒</li>
                        <li>💡 光線充足，背景簡潔</li>
                        <li>📱 保持攝影裝置穩定</li>
                    </ul>
                </div>
                """)
                
                analyze_btn = gr.Button(
                    "🚀 開始分析", 
                    variant="primary", 
                    size="lg"
                )
            
            with gr.Column(scale=2):
                # 輸出區域
                gr.HTML("<h3>📊 分析結果</h3>")
                
                # 分析結果檔案（下載用）
                output_video_file = gr.File(
                    label="📥 下載帶33個關節點標記的分析影片",
                    file_types=[".mp4"],
                    elem_classes=["output-file"]
                )
                
                # JSON 數據檔案
                output_json_file = gr.File(
                    label="📊 下載 JSON 格式數據（完整關節點數據）",
                    file_types=[".json"],
                    elem_classes=["output-file"]
                )
                
                # CSV 數據檔案
                output_csv_file = gr.File(
                    label="📈 下載 CSV 格式數據（表格數據）",
                    file_types=[".csv"],
                    elem_classes=["output-file"]
                )
                
                # 指標顯示
                metrics_output = gr.Markdown(
                    label="跑步指標",
                    value="等待分析結果...",
                    elem_classes=["metric-box"]
                )
                
                # 建議顯示
                suggestions_output = gr.Markdown(
                    label="改善建議",
                    value="上傳影片後將顯示個人化建議"
                )
                
                # 詳細計算數據顯示
                detailed_analysis = gr.DataFrame(
                    label="🔬 詳細計算數據（每幀分析，顯示前50幀）",
                    headers=["幀數", "時間(s)", "軀幹角度(°)", "左膝角度(°)", "右膝角度(°)", 
                            "手臂擺動(°)", "腳部著地", "步頻(spm)", "當幀評分"],
                    datatype=["number", "number", "number", "number", "number", 
                             "number", "str", "number", "number"],
                    wrap=True
                )
        
        # # 範例區域
        # gr.HTML("""
        # <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; margin-top: 30px;">
        #     <h3>🎯 功能特色</h3>
        #     <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 15px;">
        #         <div style="background: white; padding: 15px; border-radius: 8px;">
        #             <h4>🔍 精準檢測</h4>
        #             <p>使用 MediaPipe 技術檢測33個人體關節點，提供高精度的姿勢分析</p>
        #         </div>
        #         <div style="background: white; padding: 15px; border-radius: 8px;">
        #             <h4>📈 專業指標</h4>
        #             <p>分析軀幹角度、膝蓋抬升、腳部著地、手臂擺動等關鍵跑步指標</p>
        #         </div>
        #         <div style="background: white; padding: 15px; border-radius: 8px;">
        #             <h4>💡 個人化建議</h4>
        #             <p>根據分析結果提供針對性的改善建議，幫助提升跑步效率</p>
        #         </div>
        #         <div style="background: white; padding: 15px; border-radius: 8px;">
        #             <h4>📊 數據匯出</h4>
        #             <p>提供 JSON 和 CSV 格式的完整關節點數據，支援進一步的數據分析和研究</p>
        #         </div>
        #     </div>
        # </div>
        # """)
        
        # 設定按鈕點擊事件
        analyze_btn.click(
            fn=gradio_analyzer.analyze_video,
            inputs=[video_input],
            outputs=[output_video_file, output_json_file, output_csv_file, metrics_output, suggestions_output, detailed_analysis],
            show_progress=True
        )
    
    return interface

# 主程式
if __name__ == "__main__":
    # 創建介面
    interface = create_interface()
    
    # 啟動應用
    interface.launch(
        server_name="0.0.0.0",  # 允許外部訪問
        server_port=7860,       # Gradio 預設端口
        share=False,            # 設為 True 可生成公開連結
        debug=True,
        show_error=True
    )
