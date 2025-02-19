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
import subprocess
import numpy as np
from pathlib import Path
import tempfile
import uuid


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
            ban_emo_unk=True,      # 不禁用情感未知标签
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

def adjust_timestamps(segments, total_time, smart_distribution=True):
    """
    调整所有段落的时间戳
    - segments: 所有段落（包括已识别和未识别的）
    - total_time: 总时间
    - smart_distribution: 是否使用智能分配
    """
    if not segments:
        return []
    
    results = []
    current_time = 0
    
    if smart_distribution:
        # 智能分配：根据文本长度
        total_chars = sum(len(segment["alignedText"]) for segment in segments)
        
        for segment in segments:
            duration = (len(segment["alignedText"]) / total_chars) * total_time if total_chars > 0 else 0.1
            
            segment["timestamps"] = [current_time, current_time + duration]
            results.append(segment)
            current_time += duration
    else:
        # 平均分配：每段时间相等
        segment_duration = total_time / len(segments)
        
        for segment in segments:
            segment["timestamps"] = [current_time, current_time + segment_duration]
            results.append(segment)
            current_time += segment_duration
    
    return results

def distribute_remaining_time(unmatched_sentences, last_timestamp, total_time=1.0, smart_distribution=True):
    """
    分配时间段
    - unmatched_sentences: 未匹配的文本列表
    - last_timestamp: 最后一个已匹配文本的结束时间
    - total_time: 为未匹配文本预留的总时间（默认1秒）
    - smart_distribution: 是否使用智能分配（根据文本长度）
    """
    results = []
    
    if smart_distribution:
        # 智能分配：根据文本长度
        total_chars = sum(len(sentence) for sentence in unmatched_sentences)
        current_time = last_timestamp
        
        for sentence in unmatched_sentences:
            duration = (len(sentence) / total_chars) * total_time if total_chars > 0 else 0.1
            
            alignment_item = {
                "recognizedText": "",
                "alignedText": sentence,
                "timestamps": [
                    current_time,
                    current_time + duration
                ]
            }
            results.append(alignment_item)
            current_time += duration
    else:
        # 平均分配：每段时间相等
        segment_duration = total_time / len(unmatched_sentences) if unmatched_sentences else 0.1
        current_time = last_timestamp
        
        for sentence in unmatched_sentences:
            alignment_item = {
                "recognizedText": "",
                "alignedText": sentence,
                "timestamps": [
                    current_time,
                    current_time + segment_duration
                ]
            }
            results.append(alignment_item)
            current_time += segment_duration
    
    return results

@app.post("/api/v1/align")
async def text_alignment(
    file: UploadFile,
    text: Annotated[str, Form()],
    language: Annotated[Language, Form()] = "auto",
    auto_split: Annotated[bool, Form()] = True,
    smart_time_distribution: Annotated[bool, Form()] = True
):
    try:
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

        if len(result) == 0 or len(result[0]) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "No speech detected"}
            )

        # 获取识别结果
        timestamps = result[0][0]["timestamp"]
        recognized_text = rich_transcription_postprocess(result[0][0]["text"])

        # 处理输入文本
        input_sentences = []
        current_sentence = []

        # 分割输入文本为句子，并去除标点符号
        for char in text:
            if char in ['。', '，', '、', '！', '？', '.', ',', '!', '?']:
                sentence = ''.join(current_sentence).strip()
                if sentence:
                    input_sentences.append(sentence)  # 不添加标点
                current_sentence = []
            else:
                current_sentence.append(char)

        # 处理最后一个句子
        if current_sentence:
            sentence = ''.join(current_sentence).strip()
            if sentence:
                input_sentences.append(sentence)

        # 文本对齐处理
        current_text = []
        current_timestamps = []
        alignment_results = []

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
                        best_match = find_best_match(recognized_segment, input_sentences)
                        if best_match:
                            alignment_item = {
                                "recognizedText": recognized_segment,
                                "alignedText": best_match,  # 已经没有标点符号
                                "timestamps": [
                                    current_timestamps[0][0],
                                    current_timestamps[-1][1]
                                ]
                            }
                            alignment_results.append(alignment_item)
                            input_sentences.remove(best_match)
                    
                    current_text = []
                    current_timestamps = []

        # 处理剩余未匹配的输入文本
        if input_sentences:
            # 为未匹配文本创建结果项
            for sentence in input_sentences:
                alignment_item = {
                    "recognizedText": "",
                    "alignedText": sentence,
                    "timestamps": [0, 0]  # 临时时间戳
                }
                alignment_results.append(alignment_item)

        # 获取音频总时长
        total_duration = timestamps[-1][2] if timestamps else 30.0  # 默认30秒
        
        # 重新调整所有段落的时间戳
        alignment_results = adjust_timestamps(
            alignment_results,
            total_duration,
            smart_time_distribution
        )

        process_time = time.time() - start_time

        return {
            "success": True,
            "process_time": f"{process_time:.2f}s",
            "recognized_text": recognized_text,
            "alignment_result": alignment_results
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

def convert_to_wav(input_file_path: str, output_file_path: str) -> bool:
    """
    使用 ffmpeg 将音频转换为 wav 格式
    """
    try:
        cmd = [
            'ffmpeg', '-i', input_file_path,
            '-acodec', 'pcm_s16le',
            '-ac', '1',  # 转换为单声道
            '-ar', '16000',  # 采样率16kHz
            '-y',  # 覆盖已存在的文件
            output_file_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion error: {e.stderr.decode()}")
        return False

def split_audio(waveform: torch.Tensor, sample_rate: int, max_duration: int = 30) -> List[torch.Tensor]:
    """
    将音频切割成指定最大时长的片段
    """
    total_samples = waveform.size(-1)
    max_samples = max_duration * sample_rate
    
    # 如果音频长度小于最大时长，直接返回
    if total_samples <= max_samples:
        return [waveform]
    
    # 计算需要切割的片段数
    num_segments = (total_samples + max_samples - 1) // max_samples
    segments = []
    
    for i in range(num_segments):
        start_sample = i * max_samples
        end_sample = min((i + 1) * max_samples, total_samples)
        segment = waveform[..., start_sample:end_sample]
        segments.append(segment)
    
    return segments

def merge_results(results: List[dict]) -> dict:
    """
    合并多个识别结果
    """
    merged_full_text = ""
    merged_subtitles = []
    current_time_offset = 0.0
    
    for result in results:
        # 合并全文
        if merged_full_text:
            merged_full_text += " "
        merged_full_text += result["full_text"]
        
        # 调整时间戳并合并字幕
        for subtitle in result["subtitles"]:
            adjusted_subtitle = {
                "text": subtitle["text"],
                "timestamps": [
                    subtitle["timestamps"][0] + current_time_offset,
                    subtitle["timestamps"][1] + current_time_offset
                ]
            }
            merged_subtitles.append(adjusted_subtitle)
        
        # 更新时间偏移
        if result["subtitles"]:
            current_time_offset = merged_subtitles[-1]["timestamps"][1]
    
    return {
        "full_text": merged_full_text,
        "subtitles": merged_subtitles
    }

@app.post("/api/v1/vip/asr")
async def vip_speech_to_text(
    file: UploadFile,
    language: Annotated[Language, Form()] = "auto",
    use_itn: Annotated[bool, Form()] = True,
    output_timestamp: Annotated[bool, Form()] = True
):
    """
    VIP语音识别接口，支持更长的音频和多种格式
    - file: 音频文件(支持多种格式)
    - language: 语言选择
    - use_itn: 是否使用文本正则化
    - output_timestamp: 是否输出时间戳
    """
    try:
        start_time = time.time()
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存上传的文件
            temp_input = Path(temp_dir) / f"{uuid.uuid4()}{Path(file.filename).suffix}"
            temp_wav = Path(temp_dir) / f"{uuid.uuid4()}.wav"
            
            with open(temp_input, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # 转换为 WAV 格式
            if not convert_to_wav(str(temp_input), str(temp_wav)):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Failed to convert audio format"}
                )
            
            # 读取转换后的音频
            waveform, sample_rate = torchaudio.load(temp_wav)
            waveform = waveform.mean(0)  # 转为单声道
            
            # 切割音频
            segments = split_audio(waveform, sample_rate)
            
            # 处理每个片段
            results = []
            for segment in segments:
                # 识别
                result = m.inference(
                    data_in=segment,
                    language=language,
                    use_itn=use_itn,
                    output_timestamp=True,
                    ban_emo_unk=True,
                    fs=sample_rate,
                    **kwargs
                )
                
                if len(result) == 0 or len(result[0]) == 0:
                    continue
                
                # 处理结果
                timestamps = result[0][0]["timestamp"]
                full_text = rich_transcription_postprocess(result[0][0]["text"])
                
                # 处理时间戳和对应的文本片段
                subtitles = []
                current_text = []
                current_timestamps = []
                
                for i, ts in enumerate(timestamps):
                    if len(ts) >= 3:
                        char, start_time, end_time = ts
                        if char not in ['。', '，', '、', '！', '？', '.', ',', '!', '?']:
                            current_text.append(char)
                            current_timestamps.append([start_time, end_time])
                        
                        if char in ['。', '，', '、', '！', '？', '.', ',', '!', '?'] or i == len(timestamps) - 1:
                            sentence = ''.join(current_text).strip()
                            if sentence and current_timestamps:
                                subtitle = {
                                    "text": sentence,
                                    "timestamps": [
                                        current_timestamps[0][0],
                                        current_timestamps[-1][1]
                                    ]
                                }
                                subtitles.append(subtitle)
                                current_text = []
                                current_timestamps = []
                
                results.append({
                    "full_text": full_text,
                    "subtitles": subtitles
                })
            
            # 合并所有结果
            if not results:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No speech detected"}
                )
                
            merged_result = merge_results(results)
            process_time = time.time() - start_time
            
            return {
                "success": True,
                "process_time": f"{process_time:.2f}s",
                "full_text": merged_result["full_text"],
                "subtitles": merged_result["subtitles"]
            }

    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/v1/vip/align")
async def vip_text_alignment(
    file: UploadFile,
    text: Annotated[str, Form()],
    language: Annotated[Language, Form()] = "auto",
    auto_split: Annotated[bool, Form()] = True,
    smart_time_distribution: Annotated[bool, Form()] = True
):
    try:
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存上传的文件并转换格式
            temp_input = Path(temp_dir) / f"{uuid.uuid4()}{Path(file.filename).suffix}"
            temp_wav = Path(temp_dir) / f"{uuid.uuid4()}.wav"
            
            with open(temp_input, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # 转换为 WAV 格式（VIP特性：支持多种格式）
            if not convert_to_wav(str(temp_input), str(temp_wav)):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Failed to convert audio format"}
                )
            
            # 读取转换后的音频
            waveform, sample_rate = torchaudio.load(temp_wav)
            waveform = waveform.mean(0)  # 转为单声道
            
            # 切割音频（VIP特性：支持长音频）
            segments = split_audio(waveform, sample_rate)
            all_results = []
            recognized_text_full = ""
            
            # 处理输入文本
            input_sentences = []
            current_sentence = []
            
            # 分割输入文本为句子，并去除标点符号
            for char in text:
                if char in ['。', '，', '、', '！', '？', '.', ',', '!', '?']:
                    sentence = ''.join(current_sentence).strip()
                    if sentence:
                        input_sentences.append(sentence)  # 不添加标点
                    current_sentence = []
                else:
                    current_sentence.append(char)
            
            # 处理最后一个句子
            if current_sentence:
                sentence = ''.join(current_sentence).strip()
                if sentence:
                    input_sentences.append(sentence)
            
            # 处理每个音频片段
            for segment in segments:
                result = m.inference(
                    data_in=segment,
                    language=language,
                    use_itn=True,
                    output_timestamp=True,
                    ban_emo_unk=True,
                    fs=sample_rate,
                    **kwargs
                )
                
                if len(result) == 0 or len(result[0]) == 0:
                    continue
                
                # 处理识别结果
                recognized_text = rich_transcription_postprocess(result[0][0]["text"])
                timestamps = result[0][0]["timestamp"]
                recognized_text_full += recognized_text
                
                # 文本对齐处理
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
                                best_match = find_best_match(recognized_segment, input_sentences)
                                if best_match:
                                    alignment_item = {
                                        "recognizedText": recognized_segment,
                                        "alignedText": best_match,  # 已经没有标点符号
                                        "timestamps": [
                                            current_timestamps[0][0],
                                            current_timestamps[-1][1]
                                        ]
                                    }
                                    all_results.append(alignment_item)
                                    input_sentences.remove(best_match)
                            
                            current_text = []
                            current_timestamps = []
            
            if not all_results:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No speech detected"}
                )
            
            # 处理剩余未匹配的输入文本
            if input_sentences:
                # 为未匹配文本创建结果项
                for sentence in input_sentences:
                    alignment_item = {
                        "recognizedText": "",
                        "alignedText": sentence,
                        "timestamps": [0, 0]  # 临时时间戳
                    }
                    all_results.append(alignment_item)

            # 获取音频总时长
            total_duration = timestamps[-1][2] if timestamps else 30.0  # 默认30秒
            
            # 重新调整所有段落的时间戳
            all_results = adjust_timestamps(
                all_results,
                total_duration,
                smart_time_distribution
            )

            process_time = time.time() - start_time
            
            return {
                "success": True,
                "process_time": f"{process_time:.2f}s",
                "recognized_text": recognized_text_full,
                "alignment_result": all_results
            }

    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

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
