import cv2
import os

def video_to_frames(video_path, output_folder):
    """
    将视频的每一帧提取并保存为图像。

    参数:
    video_path (str): 输入视频文件的路径。
    output_folder (str): 保存提取帧的文件夹路径。
    """
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建文件夹: {output_folder}")

    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not video_capture.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    frame_count = 0
    while True:
        # 逐帧读取视频
        success, frame = video_capture.read()

        if not success:
            break  # 如果视频结束，则退出循环

        # 构建输出图像的文件名
        output_frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")

        # 保存当前帧为图像文件
        cv2.imwrite(output_frame_path, frame)

        frame_count += 1

    # 释放视频捕获对象
    video_capture.release()
    print(f"处理完成！总共提取了 {frame_count} 帧图像。")
    print(f"图像已保存至: {output_folder}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 设置输入视频的路径
    input_video_path = "experiment/channel-width/3-170-1.mov"  # <--- 请将此路径替换为您的视频文件路径

    # 设置保存帧图像的输出文件夹路径
    output_frames_folder = "output-frames-3-170-1"      # <--- 您可以根据需要更改此文件夹名称

    # 调用函数
    video_to_frames(input_video_path, output_frames_folder)