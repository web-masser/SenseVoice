# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from typing_extensions import Annotated
from typing import List, Optional
from enum import Enum
import torchaudio
import torch
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
import time
from fastapi.middleware.cors import CORSMiddleware


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"

model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"))
m.eval()

regex = r"<\|.*\|>"

app = FastAPI(title="SenseVoice API")

# 在创建 app 后添加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>SenseVoice API</title>
        </head>
        <body>
            <h1>SenseVoice API</h1>
            <a href='./docs'>API Documentation</a>
        </body>
    </html>
    """

@app.post("/api/v1/asr")
async def speech_to_text(
    file: UploadFile,
    language: Annotated[Language, Form()] = "auto",
    use_itn: Annotated[bool, Form()] = True,
    output_timestamp: Annotated[bool, Form()] = True
):
    """
    语音识别接口
    - file: 音频文件(wav/mp3)
    - language: 语言选择
    - use_itn: 是否使用文本正则化
    - output_timestamp: 是否输出时间戳
    """
    try:
        # 读取音频文件
        content = await file.read()
        audio_io = BytesIO(content)
        waveform, sample_rate = torchaudio.load(audio_io)
        waveform = waveform.mean(0)  # 转为单声道

        # 识别
        start_time = time.time()
        result = m.inference(
            data_in=waveform,
            language=language,
            use_itn=use_itn,
            output_timestamp=True,  # 强制开启时间戳输出
            ban_emo_unk=False,      # 不禁用情感未知标签
            fs=sample_rate,
            **kwargs
        )
        
        process_time = time.time() - start_time

        # 处理结果
        if len(result) == 0 or len(result[0]) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "No speech detected"}
            )

        # 获取时间戳和文本
        timestamps = result[0][0]["timestamp"]
        full_text = rich_transcription_postprocess(result[0][0]["text"])

        # 处理时间戳和对应的文本片段
        subtitles = []
        current_text = []
        current_timestamps = []

        for i, ts in enumerate(timestamps):
            if len(ts) >= 3:
                char, start_time, end_time = ts
                # 如果不是标点符号，则添加到当前文本
                if char not in ['。', '，', '、', '！', '？', '.', ',', '!', '?']:
                    current_text.append(char)
                    current_timestamps.append([start_time, end_time])
                
                # 遇到标点符号或最后一个字符时，保存当前句子
                if char in ['。', '，', '、', '！', '？', '.', ',', '!', '?'] or i == len(timestamps) - 1:
                    sentence = ''.join(current_text).strip()
                    if sentence and current_timestamps:
                        subtitle = {
                            "text": sentence,
                            "timestamps": [
                                current_timestamps[0][0],  # 句子开始时间
                                current_timestamps[-1][1]  # 句子结束时间
                            ]
                        }
                        subtitles.append(subtitle)
                        # 重置状态，准备下一句
                        current_text = []
                        current_timestamps = []

        return {
            "success": True,
            "process_time": f"{process_time:.2f}s",
            "full_text": full_text,
            "subtitles": subtitles
        }

    except Exception as e:
        print(f"Error: {str(e)}")  # 添加错误日志
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def format_time(seconds):
    """将秒数转换为 SRT 格式的时间戳"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millisecs = int((secs - int(secs)) * 1000)
    secs = int(secs)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def adjust_timestamps(recognized_segments, input_segments, total_duration):
    """智能调整时间戳以匹配输入文本段落数量"""
    if not input_segments:
        return []
    
    # 计算输入文本段落的长度权重
    input_lengths = [len(seg) for seg in input_segments]
    total_input_length = sum(input_lengths)
    input_weights = [length / total_input_length for length in input_lengths]
    
    result = []
    
    # 1. 先保留所有已识别的段落
    for i in range(len(recognized_segments)):
        result.append({
            "recognizedText": recognized_segments[i]["recognizedText"],
            "alignedText": recognized_segments[i]["alignedText"],
            "timestamps": recognized_segments[i]["timestamps"]
        })
    
    # 2. 计算剩余时间和剩余段落
    last_end_time = recognized_segments[-1]["timestamps"][1] if recognized_segments else 0
    remaining_duration = total_duration - last_end_time
    remaining_segments = input_segments[len(recognized_segments):]
    
    if not remaining_segments:
        return result
    
    # 3. 计算剩余段落的权重
    remaining_lengths = [len(seg) for seg in remaining_segments]
    remaining_total_length = sum(remaining_lengths)
    remaining_weights = [length / remaining_total_length for length in remaining_lengths]
    
    # 4. 为剩余段落分配时间
    current_time = last_end_time
    
    for i, segment in enumerate(remaining_segments):
        # 计算当前段落应该分配的时长
        if i == len(remaining_segments) - 1:
            # 最后一个段落使用所有剩余时间
            duration = total_duration - current_time
        else:
            # 根据文本长度权重分配时间
            duration = remaining_duration * remaining_weights[i]
        
        # 添加新的对齐结果
        result.append({
            "recognizedText": "",  # 识别文本为空
            "alignedText": segment,
            "timestamps": [
                current_time,
                current_time + duration
            ]
        })
        
        current_time += duration
    
    return result

@app.post("/api/v1/align")
async def text_alignment(
    file: UploadFile,
    text: Annotated[str, Form()],
    language: Annotated[Language, Form()] = "auto",
    auto_split: Annotated[bool, Form()] = True
):
    """
    文本对齐打轴接口
    - file: 音频文件
    - text: 待对齐的文本内容
    - language: 语言选择
    - auto_split: 是否自动分段
    """
    try:
        # 使用 perf_counter 获取更精确的时间
        start_time = time.time()
      
        
        
        
        # 读取音频文件
        content = await file.read()
        audio_io = BytesIO(content)
        waveform, sample_rate = torchaudio.load(audio_io)
        waveform = waveform.mean(0)

        # 先进行语音识别
        result = m.inference(
            data_in=waveform,
            language=language,
            use_itn=True,
            output_timestamp=True,
            ban_emo_unk=False,
            fs=sample_rate,
            **kwargs
        )

        process_time = time.time() - start_time

        if len(result) == 0 or len(result[0]) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "No speech detected"}
            )

        # 获取识别结果
        timestamps = result[0][0]["timestamp"]
        recognized_text = rich_transcription_postprocess(result[0][0]["text"])

        # 获取音频总时长
        total_duration = timestamps[-1][2]  # 使用最后一个时间戳的结束时间
        
        # 处理输入文本，按标点符号分段
        input_segments = re.split(r'[。，、！？.,!?]', text)
        input_segments = [s.strip() for s in input_segments if s.strip()]
        remaining_segments = input_segments.copy()  # 创建副本以保留原始输入
        
        # 先进行正常的对齐
        alignment_result = []
        current_text = []
        current_timestamps = []

        for i, ts in enumerate(timestamps):
            if len(ts) >= 3:
                char, start_time, end_time = ts
                if char not in ['。', '，', '、', '！', '？', '.', ',', '!', '?']:
                    current_text.append(char)
                    current_timestamps.append([start_time, end_time])
                
                if char in ['。', '，', '、', '！', '？', '.', ',', '!', '?'] or i == len(timestamps) - 1:
                    recognized_segment = ''.join(current_text).strip()
                    if recognized_segment:
                        # 找到最匹配的输入文本段
                        best_match = find_best_match(recognized_segment, remaining_segments)
                        if best_match:
                            alignment_result.append({
                                "recognizedText": recognized_segment,
                                "alignedText": best_match,
                                "timestamps": [
                                    current_timestamps[0][0],
                                    current_timestamps[-1][1]
                                ]
                            })
                            remaining_segments.remove(best_match)  # 从剩余段落中移除
                    
                    current_text = []
                    current_timestamps = []

        # 如果还有未匹配的输入段落，进行智能调整
        if remaining_segments:
            # 获取已匹配段落的最后时间戳
            last_end_time = alignment_result[-1]["timestamps"][1] if alignment_result else 0
            remaining_duration = total_duration - last_end_time
            
            # 计算剩余段落的权重
            remaining_lengths = [len(seg) for seg in remaining_segments]
            remaining_total_length = sum(remaining_lengths)
            remaining_weights = [length / remaining_total_length for length in remaining_lengths]
            
            # 为剩余段落分配时间
            current_time = last_end_time
            
            for i, segment in enumerate(remaining_segments):
                # 计算当前段落应该分配的时长
                if i == len(remaining_segments) - 1:
                    duration = total_duration - current_time
                else:
                    duration = remaining_duration * remaining_weights[i]
                
                # 添加新的对齐结果
                alignment_result.append({
                    "recognizedText": "",  # 识别文本为空
                    "alignedText": segment,
                    "timestamps": [
                        current_time,
                        current_time + duration
                    ]
                })
                
                current_time += duration

        return {
            "success": True,
            "process_time": f"{process_time:.2f}s",  # 只返回数值，不加单位
            "recognized_text": recognized_text,
            "alignment_result": alignment_result
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def find_best_match(recognized_text, input_segments, min_similarity=0.3):
    """使用改进的相似度匹配算法找到最匹配的文本段"""
    if not input_segments:
        return None
    
    def clean_text(text):
        """清理文本，去除标点和空格"""
        return re.sub(r'[^\w\u4e00-\u9fff]', '', text)
    
    def calculate_similarity(text1, text2):
        """计算两段文本的相似度"""
        # 使用编辑距离和最长公共子序列结合的方式
        text1 = clean_text(text1)
        text2 = clean_text(text2)
        
        if not text1 or not text2:
            return 0
            
        # 计算最长公共子序列
        def lcs_length(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        lcs = lcs_length(text1, text2)
        max_len = max(len(text1), len(text2))
        min_len = min(len(text1), len(text2))
        
        if max_len == 0:
            return 0
            
        # 结合长度比例和LCS比例
        length_ratio = min_len / max_len
        lcs_ratio = lcs / max_len
        
        return (length_ratio * 0.4 + lcs_ratio * 0.6)
    
    # 计算所有候选文本的相似度
    similarities = [calculate_similarity(recognized_text, segment) 
                   for segment in input_segments]
    
    if not similarities:
        return None
        
    max_similarity = max(similarities)
    best_index = similarities.index(max_similarity)
    
    # 如果最佳匹配的相似度太低，尝试合并相邻段落
    if max_similarity < min_similarity and len(input_segments) > 1:
        merged_segments = []
        merged_similarities = []
        
        # 尝试合并相邻的两个段落
        for i in range(len(input_segments) - 1):
            merged_text = input_segments[i] + input_segments[i + 1]
            similarity = calculate_similarity(recognized_text, merged_text)
            merged_segments.append((i, merged_text))
            merged_similarities.append(similarity)
        
        if merged_similarities:
            max_merged_similarity = max(merged_similarities)
            if max_merged_similarity > max_similarity:
                best_merged_index = merged_similarities.index(max_merged_similarity)
                start_index = merged_segments[best_merged_index][0]
                # 合并相邻段落并从输入段落列表中移除
                merged_text = merged_segments[best_merged_index][1]
                input_segments.pop(start_index + 1)
                input_segments[start_index] = merged_text
                return merged_text
    
    return input_segments[best_index] if max_similarity >= min_similarity else None

@app.get("/api/v1/languages")
async def get_supported_languages():
    """获取支持的语言列表"""
    return {
        "languages": [
            {"code": "zh", "name": "中文"},
            {"code": "en", "name": "英文"},
            {"code": "yue", "name": "粤语"},
            {"code": "ja", "name": "日语"},
            {"code": "ko", "name": "韩语"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    # 配置服务器启动参数
    uvicorn.run(
        "api:app",
        host="0.0.0.0",  # 允许外部访问
        port=5332,       # 指定端口
        reload=True,     # 开发模式下启用热重载
        workers=1        # 工作进程数
    )
