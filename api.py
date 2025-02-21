# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
from fastapi import FastAPI, File, Form, UploadFile, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response
from typing_extensions import Annotated
from typing import List, Optional, Dict
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
import ssl
import json
import asyncio
import base64
import difflib
import cv2
from PIL import Image, ImageDraw, ImageFont
from starlette.websockets import WebSocketDisconnect

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

# 存储WebSocket连接
websocket_connections: Dict[str, WebSocket] = {}

# 添加 WebSocket 连接存储
progress_connections: Dict[str, WebSocket] = {}

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

        # 处理识别结果
        if len(result) > 0 and len(result[0]) > 0:
            timestamps = result[0][0]["timestamp"]
            recognized_text = rich_transcription_postprocess(result[0][0]["text"])
            
            # 先把识别的文本按标点符号分段
            recognized_sentences = []
            current_text = []
            current_timestamps = []
            
            # 按字符处理识别文本
            for i, ts in enumerate(timestamps):
                if len(ts) >= 3:
                    char, start_time, end_time = ts
                    if char not in ['。', '，', '、', '！', '？', '.', ',', '!', '?']:
                        current_text.append(char)
                        current_timestamps.append([start_time, end_time])
                    
                    # 当遇到标点或是最后一个字符时，保存当前句子
                    if (char in ['。', '，', '、', '！', '？', '.', ',', '!', '?'] or i == len(timestamps) - 1) and current_text:
                        recognized_sentences.append({
                            "text": ''.join(current_text),
                            "timestamps": [
                                current_timestamps[0][0],
                                current_timestamps[-1][1]
                            ]
                        })
                        current_text = []
                        current_timestamps = []
            
            # 分割输入文本（按逗号分割）
            input_sentences = [s.strip() for s in text.split(',') if s.strip()]
            
            # 在计算总时长之前，先计算 max_length
            # 使用较长的列表的长度作为循环次数
            max_length = max(len(recognized_sentences), len(input_sentences))

            # 获取总时长
            total_time = 0
            if recognized_sentences:
                last_segment = recognized_sentences[-1]
                # timestamps 是 [start_time, end_time] 格式
                total_time = last_segment["timestamps"][1]  # 使用结束时间

            # 计算每个段落应该分配的时间
            segment_duration = total_time / max_length if max_length > 0 else 30

            # 创建对齐结果
            alignment_results = []

            # 创建所有对齐项
            for i in range(max_length):
                # 如果有识别文本，使用它的时间戳；否则计算一个合理的时间戳
                if i < len(recognized_sentences):
                    recognized = recognized_sentences[i]
                    timestamps = recognized["timestamps"]
                else:
                    # 为多余的段落计算时间戳
                    start_time = i * segment_duration
                    end_time = (i + 1) * segment_duration
                    timestamps = [start_time, end_time]
                    recognized = {"text": "", "timestamps": timestamps}
                
                # 如果有输入文本，使用它；否则使用空字符串
                input_text = input_sentences[i] if i < len(input_sentences) else ""
                
                alignment_item = {
                    "recognizedText": recognized["text"],
                    "alignedText": input_text,
                    "timestamps": timestamps
                }
                alignment_results.append(alignment_item)
            
            process_time = time.time() - start_time
            
            return {
                "success": True,
                "process_time": f"{process_time:.2f}s",
                "recognized_text": recognized_text,
                "alignment_result": alignment_results
            }
        
        return JSONResponse(
            status_code=400,
            content={"error": "No speech detected"}
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def find_best_match(recognized_text: str, input_sentences: list) -> str:
    """
    使用序列匹配算法找到最匹配的文本
    """
    if not input_sentences:
        return None
    
    # 移除表情符号和标点
    recognized_text = re.sub(r'[🎼😊。，、！？.,!?]', '', recognized_text)
    
    # 使用序列匹配找到最佳匹配
    matches = []
    for sentence in input_sentences:
        ratio = difflib.SequenceMatcher(None, recognized_text, sentence).ratio()
        matches.append((ratio, sentence))
    
    # 找到匹配度最高的句子
    best_match = max(matches, key=lambda x: x[0])
    
    # 如果匹配度太低，返回None
    return best_match[1] if best_match[0] > 0.6 else None

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


@app.websocket("/api/v1/vip/asr/ws")
async def vip_speech_to_text_ws(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # 接收音频数据
        data = await websocket.receive_bytes()
        
        # 创建临时目录处理音频
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存接收到的音频数据
            temp_input = Path(temp_dir) / f"{uuid.uuid4()}.wav"
            with open(temp_input, "wb") as f:
                f.write(data)
            
            # 读取音频
            waveform, sample_rate = torchaudio.load(temp_input)
            waveform = waveform.mean(0)  # 转为单声道
            
            # 切割音频
            segments = split_audio(waveform, sample_rate)
            total_segments = len(segments)
            
            # 处理每个片段
            all_results = []
            
            # 发送总片段数
            await websocket.send_json({
                "type": "start",
                "total_segments": total_segments
            })
            
            # 串行处理每个片段
            for i, segment in enumerate(segments, 1):
                # 发送当前正在处理的片段信息
                await websocket.send_json({
                    "type": "processing",
                    "current_segment": i,
                    "total_segments": total_segments,
                    "message": f"正在处理第 {i}/{total_segments} 段..."
                })
                
                # 识别当前片段
                result = m.inference(
                    data_in=segment,
                    language="auto",
                    use_itn=True,
                    output_timestamp=True,
                    ban_emo_unk=True,
                    fs=sample_rate,
                    **kwargs
                )
                
                if len(result) > 0 and len(result[0]) > 0:
                    # 处理识别结果
                    timestamps = result[0][0]["timestamp"]
                    text = rich_transcription_postprocess(result[0][0]["text"])
                    
                    # 处理字幕
                    subtitles = []
                    current_text = []
                    current_timestamps = []
                    
                    for j, ts in enumerate(timestamps):
                        if len(ts) >= 3:
                            char, start_time, end_time = ts
                            if char not in ['。', '，', '、', '！', '？', '.', ',', '!', '?']:
                                current_text.append(char)
                                current_timestamps.append([start_time, end_time])
                            
                            if char in ['。', '，', '、', '！', '？', '.', ',', '!', '?'] or j == len(timestamps) - 1:
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
                    
                    segment_result = {
                        "text": text,
                        "subtitles": subtitles
                    }
                    all_results.append(segment_result)
                    
                    # 发送当前片段处理完成的结果
                    await websocket.send_json({
                        "type": "segment_complete",
                        "current_segment": i,
                        "total_segments": total_segments,
                        "segment_result": segment_result
                    })
                
                # 等待一小段时间，确保串行处理
                await asyncio.sleep(0.1)
            
            # 发送所有处理完成的消息
            await websocket.send_json({
                "type": "complete",
                "results": all_results
            })
            
    except Exception as e:
        # 发送错误消息
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        await websocket.close()

@app.websocket("/api/v1/vip/align/ws")
async def vip_text_alignment_ws(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # 1. 接收配置信息
        config = await websocket.receive_json()
        if config['type'] != 'config':
            raise ValueError("Expected config message first")
            
        text = config.get("text")
        language = config.get("language", "auto")
        smart_time_distribution = config.get("smart_time_distribution", True)
        
        # 2. 接收音频数据
        audio_data = await websocket.receive_bytes()
        
        # 3. 处理音频
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = Path(temp_dir) / f"{uuid.uuid4()}.wav"
            with open(temp_input, "wb") as f:
                f.write(audio_data)
            
            # 读取音频
            waveform, sample_rate = torchaudio.load(temp_input)
            waveform = waveform.mean(0)  # 转为单声道
            
            # 切割音频
            segments = split_audio(waveform, sample_rate)
            total_segments = len(segments)
            
            # 发送开始消息
            await websocket.send_json({
                "type": "start",
                "total_segments": total_segments
            })
            
            # 第一阶段：处理所有音频片段，只返回进度
            all_recognized_segments = []
            
            for i, segment in enumerate(segments, 1):
                # 发送处理进度
                await websocket.send_json({
                    "type": "processing",
                    "current_segment": i,
                    "total_segments": total_segments,
                    "message": f"正在处理第 {i}/{total_segments} 段音频..."
                })
                
                # 识别当前片段
                result = m.inference(
                    data_in=segment,
                    language=language,
                    use_itn=True,
                    output_timestamp=True,
                    ban_emo_unk=True,
                    fs=sample_rate,
                    **kwargs
                )
                
                if len(result) > 0 and len(result[0]) > 0:
                    timestamps = result[0][0]["timestamp"]
                    recognized_text = rich_transcription_postprocess(result[0][0]["text"])
                    
                    # 保存识别结果
                    all_recognized_segments.append({
                        "text": recognized_text,
                        "timestamps": timestamps,
                        "segment_index": i
                    })
                
                await asyncio.sleep(0.1)
            
            # 第二阶段：处理文本对齐
            await websocket.send_json({
                "type": "aligning",
                "message": "音频处理完成，正在对齐文本..."
            })
            
            # 先把识别的完整文本按标点符号分段
            full_text = "".join(seg["text"] for seg in all_recognized_segments)
            recognized_sentences = []
            current_sentence = []
            current_start_time = 0
            
            # 遍历所有识别片段的时间戳
            for segment in all_recognized_segments:
                timestamps = segment["timestamps"]
                for i, ts in enumerate(timestamps):
                    if len(ts) >= 3:
                        char, start_time, end_time = ts
                        if char not in ['。', '，', '、', '！', '？', '.', ',', '!', '?']:
                            current_sentence.append(char)
                            if len(current_sentence) == 1:  # 如果是句子的第一个字符
                                current_start_time = start_time + (segment["segment_index"]-1) * 30
                        
                        # 当遇到标点或是最后一个字符时，保存当前句子
                        if (char in ['。', '，', '、', '！', '？', '.', ',', '!', '?'] or i == len(timestamps) - 1) and current_sentence:
                            recognized_sentences.append({
                                "text": ''.join(current_sentence),
                                "timestamps": [
                                    current_start_time,
                                    end_time + (segment["segment_index"]-1) * 30
                                ]
                            })
                            current_sentence = []
            
            # 分割输入文本（按逗号分割）
            input_sentences = [s.strip() for s in text.split(',') if s.strip()]
            
            # 在计算总时长之前，先计算 max_length
            # 使用较长的列表的长度作为循环次数
            max_length = max(len(recognized_sentences), len(input_sentences))

            # 获取总时长
            total_time = 0
            if recognized_sentences:
                last_segment = recognized_sentences[-1]
                # timestamps 是 [start_time, end_time] 格式
                total_time = last_segment["timestamps"][1]  # 使用结束时间

            # 计算每个段落应该分配的时间
            segment_duration = total_time / max_length if max_length > 0 else 30

            # 创建对齐结果
            alignment_results = []

            # 创建所有对齐项
            for i in range(max_length):
                # 如果有识别文本，使用它的时间戳；否则计算一个合理的时间戳
                if i < len(recognized_sentences):
                    recognized = recognized_sentences[i]
                    timestamps = recognized["timestamps"]
                else:
                    # 为多余的段落计算时间戳
                    start_time = i * segment_duration
                    end_time = (i + 1) * segment_duration
                    timestamps = [start_time, end_time]
                    recognized = {"text": "", "timestamps": timestamps}
                
                # 如果有输入文本，使用它；否则使用空字符串
                input_text = input_sentences[i] if i < len(input_sentences) else ""
                
                alignment_item = {
                    "recognizedText": recognized["text"],
                    "alignedText": input_text,
                    "timestamps": timestamps
                }
                alignment_results.append(alignment_item)
            
            # 发送完成消息
            await websocket.send_json({
                "type": "complete",
                "results": alignment_results,
                "recognized_text": full_text
            })
            
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        await websocket.close()

@app.websocket("/api/v1/merge-subtitle/progress/{task_id}")
async def merge_subtitle_progress(websocket: WebSocket, task_id: str):
    print(f"WebSocket connection attempt for task {task_id}")
    await websocket.accept()
    try:
        progress_connections[task_id] = websocket
        print(f"WebSocket connected for task {task_id}")
        await websocket.send_json({"progress": 0})
        
        try:
            while True:
                await websocket.receive_text()
                await asyncio.sleep(0.1)  # 添加小延迟防止过快循环
        except WebSocketDisconnect:
            print(f"WebSocket disconnected normally for task {task_id}")
        except Exception as e:
            print(f"WebSocket error for task {task_id}: {e}")
    finally:
        if task_id in progress_connections:
            del progress_connections[task_id]
            print(f"Cleaned up connection for task {task_id}")

@app.post("/api/v1/merge-subtitle")
async def merge_subtitle(
    video: UploadFile,
    srt: UploadFile,
    position: Annotated[str, Form()] = "bottom",
    font_size: Annotated[int, Form()] = 24,
    x_offset: Annotated[int, Form()] = 0,
    y_offset: Annotated[int, Form()] = 0,
    font_color: Annotated[str, Form()] = "#FFFFFF",
    outline_color: Annotated[str, Form()] = "#000000",
    outline_width: Annotated[float, Form()] = 2.0,
    task_id: Annotated[str, Form()] = None
):
    try:
        task_id = task_id or str(uuid.uuid4())
        print(f"Starting merge task with ID: {task_id}")
        
        # 创建临时文件夹
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # 保存上传的文件，移除文件名中的特殊字符
        safe_video_name = ''.join(c for c in video.filename if c.isalnum() or c in '._-')
        video_path = temp_dir / f"{uuid.uuid4()}_{safe_video_name}"
        srt_path = temp_dir / f"{uuid.uuid4()}.srt"
        output_path = temp_dir / f"output_{uuid.uuid4()}.mp4"
        final_output = temp_dir / f"final_{uuid.uuid4()}.mp4"
        
        try:
            # 写入视频和字幕文件
            video_content = await video.read()
            srt_content = await srt.read()
            
            with open(str(video_path), "wb") as f:
                f.write(video_content)
            with open(str(srt_path), "wb") as f:
                f.write(srt_content)

            # 读取字幕文件
            def parse_srt(srt_path):
                with open(str(srt_path), 'r', encoding='utf-8') as f:
                    content = f.read()
                
                subtitles = []
                blocks = content.strip().split('\n\n')
                for block in blocks:
                    lines = block.split('\n')
                    if len(lines) >= 3:
                        time_line = lines[1]
                        start_time, end_time = time_line.split(' --> ')
                        text = ' '.join(lines[2:])
                        
                        # 转换时间为秒
                        def time_to_seconds(t):
                            h, m, s = t.split(':')
                            s, ms = s.split(',')
                            return float(h) * 3600 + float(m) * 60 + float(s) + float(ms) / 1000
                        
                        subtitles.append({
                            'start': time_to_seconds(start_time),
                            'end': time_to_seconds(end_time),
                            'text': text
                        })
                return subtitles

            # 读取字幕
            subtitles = parse_srt(srt_path)

            # 读取视频
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception("无法打开视频文件")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 加载字体
            font_path = "C:\\Windows\\Fonts\\msyh.ttc"  # 使用完整路径
            if not os.path.exists(font_path):
                font_path = "C:\\Windows\\Fonts\\arial.ttf"  # 备选字体
            font = ImageFont.truetype(font_path, font_size)

            # 处理颜色格式
            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

            font_rgb = hex_to_rgb(font_color)
            outline_rgb = hex_to_rgb(outline_color)

            # 使用 H.264 编码器
            temp_output = temp_dir / f"temp_{uuid.uuid4()}.mp4"
            
            # 先用 OpenCV 处理帧
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
            
            # 处理每一帧
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 计算并发送进度
                progress = int((frame_count / total_frames) * 100)
                if frame_count % 30 == 0 and task_id in progress_connections:
                    try:
                        print(f"Sending progress {progress}% for task {task_id}")
                        await progress_connections[task_id].send_json({
                            "progress": progress,
                            "frame": frame_count,
                            "total": total_frames
                        })
                        # 添加小延迟，让前端有时间处理
                        await asyncio.sleep(0.01)
                    except Exception as e:
                        print(f"Error sending progress for task {task_id}: {e}")

                current_time = frame_count / fps
                
                # 查找当前时间的字幕
                current_text = ""
                for sub in subtitles:
                    if sub['start'] <= current_time <= sub['end']:
                        current_text = sub['text']
                        break

                if current_text:
                    # 将 OpenCV 图像转换为 PIL 图像
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(pil_img)

                    # 计算文本大小
                    text_bbox = draw.textbbox((0, 0), current_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    # 计算文本位置
                    x = (width - text_width) // 2 + x_offset
                    if position == "bottom":
                        y = height - text_height - 50 + y_offset
                    elif position == "top":
                        y = 50 + y_offset
                    else:  # middle
                        y = (height - text_height) // 2 + y_offset

                    # 绘制描边
                    for dx, dy in [(j, i) for i in range(-int(outline_width), int(outline_width) + 1)
                                        for j in range(-int(outline_width), int(outline_width) + 1)]:
                        draw.text((x + dx, y + dy), current_text, font=font, fill=outline_rgb)

                    # 绘制文本
                    draw.text((x, y), current_text, font=font, fill=font_rgb)

                    # 转回 OpenCV 格式
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                # 写入帧
                out.write(frame)

                # 每处理一定数量的帧后让出控制权
                if frame_count % 10 == 0:
                    await asyncio.sleep(0)

                frame_count += 1

            # 释放资源
            cap.release()
            out.release()

            # 使用 ffmpeg 重新编码，确保使用兼容的编码器
            cmd = [
                'ffmpeg',
                '-i', str(temp_output),
                '-i', str(video_path),
                '-c:v', 'libx264',  # 使用 H.264 编码
                '-preset', 'medium',
                '-crf', '23',       # 控制视频质量
                '-c:a', 'aac',      # 音频编码
                '-strict', 'experimental',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-y',
                str(final_output)
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise Exception(f"FFmpeg error: {process.stderr}")

            # 读取最终输出文件
            with open(str(final_output), "rb") as f:
                video_data = f.read()

            # 发送100%进度
            if task_id in progress_connections:
                try:
                    await progress_connections[task_id].send_json({
                        "progress": 100,
                        "frame": total_frames,
                        "total": total_frames
                    })
                except Exception as e:
                    print(f"Error sending final progress: {e}")

            # 返回处理后的视频
            return Response(
                content=video_data,
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f"attachment; filename=output_{safe_video_name}"
                }
            )
            
        finally:
            # 清理临时文件
            for file in [video_path, srt_path, output_path, final_output]:
                try:
                    if file.exists():
                        file.unlink()
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def get_video_info(video_path: str) -> dict:
    """获取视频信息"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFprobe error: {result.stderr}")
    
    info = json.loads(result.stdout)
    return info.get('streams', [{}])[0]

if __name__ == "__main__":
    import uvicorn
    
    # SSL配置
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(
        certfile="mznpy.com.pem",
        keyfile="mznpy.com.key"
    )
    
    # 配置服务器启动参数
    uvicorn.run(
        "api:app",
        host="0.0.0.0",      # 允许外部访问
        port=5332,           # 指定端口
        workers=1,           # 工作进程数
        ssl_keyfile="mznpy.com.key",    # SSL密钥文件
        ssl_certfile="mznpy.com.pem",   # SSL证书文件
    )
