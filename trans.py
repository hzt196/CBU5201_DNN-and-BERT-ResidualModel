import whisper
from imageio_ffmpeg import get_ffmpeg_exe
import subprocess
import numpy as np
import os

# 获取 FFmpeg 可执行文件路径
ffmpeg_path = get_ffmpeg_exe()

# 检查 FFmpeg 是否可用
try:
    subprocess.run([ffmpeg_path, "-version"], check=True)
    print("FFmpeg 可用！路径:", ffmpeg_path)
except Exception as e:
    print("FFmpeg 检查失败:", e)
    exit(1)

# 自定义音频加载函数，使用指定的 FFmpeg
def load_audio_custom(file: str):
    """
    使用 imageio-ffmpeg 提供的 FFmpeg 加载音频，并返回 numpy 数组。
    """
    cmd = [
        ffmpeg_path, "-i", file, "-f", "wav", "-ac", "1", "-ar", "16000", "-"
    ]  # 输出单声道、16kHz 的 WAV 格式
    try:
        process = subprocess.run(cmd, capture_output=True, check=True)
        audio_data = process.stdout  # 获取音频数据（bytes）
        
        # 将音频字节流转换为 numpy 数组
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # 归一化音频数据
        audio_np = audio_np.astype(np.float32) / 32768.0  # 将 int16 转为 float32，并归一化
        
        return audio_np
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 转换失败: {e}")
        exit(1)

# 替换 whisper 的默认音频加载逻辑
from whisper.audio import load_audio
whisper.audio.load_audio = load_audio_custom

# Whisper 模型加载
print("加载 Whisper 模型...")
model = whisper.load_model("base")  # 可选 tiny, base, small, medium, large

# 定义音频文件目录和输出目录
audio_dir = "Deception-main\CBU0521DD_stories"  # 替换为存放音频的目录
output_dir = "Deception-main\output"  # 替换为存放结果文本的目录
os.makedirs(output_dir, exist_ok=True)  # 如果输出目录不存在则创建

# 遍历音频文件目录
for file_name in os.listdir(audio_dir):
    if file_name.endswith(".wav"):
        audio_path = os.path.join(audio_dir, file_name)
        output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.txt")
        
        print(f"处理文件: {audio_path}")

        # 语音转文字
        result = model.transcribe(audio_path, fp16=False)  # 如果有 GPU，设置 fp16=True

        # 保存文字到文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"结果保存到: {output_path}")

print("处理完成！")
