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
from funasr import AutoModel
import ffmpeg
import traceback
import aiofiles

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

def adjust_timestamps(segments, total_time):
    """
    重新调整所有段落的时间戳
    - segments: 所有段落（包括识别文本和对齐文本）
    - total_time: 音频总时长
    """
    if not segments:
        return []
    
    # 计算所有文本的总字符数（优先使用对齐文本）
    total_chars = sum(len(segment["alignedText"] or segment["recognizedText"]) for segment in segments)
    
    results = []
    current_time = 0
    
    # 根据文本长度比例分配时间
    for segment in segments:
        # 优先使用对齐文本的长度
        text_length = len(segment["alignedText"] or segment["recognizedText"])
        
        # 计算该段落应占用的时间长度
        duration = (text_length / total_chars) * total_time if total_chars > 0 else 0.1
        
        # 更新时间戳
        segment["timestamps"] = [current_time, current_time + duration]
        results.append(segment)
        current_time += duration
    
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
        print("开始处理新的 WebSocket 请求...")
        print("等待接收音频数据...")
        
        data = await websocket.receive_bytes()
        print(f"接收到音频数据，大小: {len(data)} bytes")
        
        # 创建临时目录存储音频文件
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = Path(temp_dir) / f"original_{uuid.uuid4()}"
            temp_wav = Path(temp_dir) / f"{uuid.uuid4()}.wav"
            
            # 写入接收到的数据
            async with aiofiles.open(temp_input, "wb") as f:
                await f.write(data)
            
            try:
                # 修改 ffprobe 命令执行方式
                probe_cmd = [
                    'ffprobe', 
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_format',
                    '-show_streams',
                    str(temp_input)
                ]
                probe_result = subprocess.run(
                    probe_cmd, 
                    capture_output=True, 
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                if not probe_result.stdout:
                    raise ValueError("无法读取文件格式信息")
                
                try:
                    format_info = json.loads(probe_result.stdout)
                except json.JSONDecodeError:
                    raise ValueError("文件格式信息解析失败")
                
                # 检查是否包含音频流
                has_audio = any(stream['codec_type'] == 'audio' 
                              for stream in format_info.get('streams', []))
                
                if not has_audio:
                    await websocket.send_json({
                        "type": "error",
                        "message": "文件中未检测到音频内容"
                    })
                    return
                
                # 提取/转换音频为 WAV 格式
                cmd = [
                    'ffmpeg', '-i', str(temp_input),
                    '-vn',  # 去除视频流
                    '-acodec', 'pcm_s16le',
                    '-ac', '1',  # 转换为单声道
                    '-ar', '16000',  # 采样率16kHz
                    '-y',  # 覆盖已存在的文件
                    str(temp_wav)
                ]
                
                # 执行 ffmpeg 命令
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                if process.returncode != 0:
                    raise ValueError(f"音频转换失败: {process.stderr}")
                
                # 读取转换后的 WAV 文件
                waveform, sample_rate = torchaudio.load(temp_wav)
                waveform = waveform.mean(0)  # 转为单声道
                
                # 切割音频
                segments = split_audio(waveform, sample_rate)
                total_segments = len(segments)
                
                print(f"音频片段数量: {total_segments}")
                
                all_results = []
                current_time_offset = 0  # 添加时间偏移量
                
                for i, segment in enumerate(segments, 1):
                    print(f"\n处理第 {i}/{total_segments} 段:")
                    
                    result = m.inference(
                        data_in=segment,
                        language="auto",
                        use_itn=True,
                        output_timestamp=True,
                        ban_emo_unk=True,
                        fs=sample_rate,
                        **kwargs
                    )

                    if result and len(result) > 0 and len(result[0]) > 0:
                        text = result[0][0]["text"]
                        
                        subtitles = []
                        if "timestamp" in result[0][0]:
                            timestamps = result[0][0]["timestamp"]
                            
                            current_text = []
                            current_timestamps = []
                            last_end_time = 0
                            
                            for j, ts in enumerate(timestamps):
                                if len(ts) >= 3:
                                    char, start_time, end_time = ts
                                    # 添加时间偏移
                                    start_time += current_time_offset
                                    end_time += current_time_offset
                                    
                                    if char not in ['。', '，', '、', '！', '？', '.', ',', '!', '?']:
                                        current_text.append(char)
                                        current_timestamps.append([start_time, end_time])
                                        last_end_time = end_time
                                    elif current_text:
                                        sentence = ''.join(current_text).strip()
                                        if sentence and current_timestamps:
                                            subtitle = {
                                                "text": sentence,
                                                "timestamps": [
                                                    current_timestamps[0][0],
                                                    end_time
                                                ]
                                            }
                                            subtitles.append(subtitle)
                                            current_text = []
                                            current_timestamps = []
                                            last_end_time = end_time
                        
                        segment_result = {
                            "text": text,
                            "subtitles": subtitles
                        }
                        all_results.append(segment_result)
                        
                        # 更新时间偏移量
                        current_time_offset = last_end_time
                        
                        # 发送进度更新
                        await websocket.send_json({
                            "type": "segment_complete",
                            "current_segment": i,
                            "total_segments": total_segments,
                            "segment_result": segment_result
                        })
                
                # 发送最终结果
                await websocket.send_json({
                    "type": "complete",
                    "results": all_results
                })
                
            except Exception as e:
                print(f"处理错误: {str(e)}")
                traceback.print_exc()
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except Exception as e:
        print(f"发生错误: {str(e)}")
        traceback.print_exc()
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        print("WebSocket 连接关闭")
        await websocket.close()

@app.websocket("/api/v1/vip/align/ws")
async def vip_text_alignment_ws(websocket: WebSocket):
    print("\n=== 开始新的对齐请求 ===")
    await websocket.accept()
    
    try:
        # 1. 接收配置信息
        config = await websocket.receive_json()
        print(f"\n配置信息: {config}")
        
        if config['type'] != 'config':
            raise ValueError("Expected config message first")
        
        # 获取对齐文本并按标点符号分段
        align_text = config.get("text", "")
        print(f"\n对齐文本: {align_text}")
        
        # 使用正则表达式分割文本
        pattern = r'[，,。.！!？?、]'
        align_segments = [seg.strip() for seg in re.split(pattern, align_text) if seg.strip()]
        print(f"\n对齐文本分段: {align_segments}")
        print(f"对齐文本段落数: {len(align_segments)}")
        
        # 2. 接收并处理音频
        audio_data = await websocket.receive_bytes()
        print(f"\n接收到音频数据: {len(audio_data)} bytes")
        
        # 3. 音频转换和识别
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存原始文件
            original_file = Path(temp_dir) / f"original_{uuid.uuid4()}"
            with open(original_file, "wb") as f:
                f.write(audio_data)
            
            # 使用 ffprobe 检测文件格式
            try:
                probe_cmd = [
                    'ffprobe', 
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_format',
                    '-show_streams',
                    str(original_file)
                ]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                format_info = json.loads(probe_result.stdout)
                
                # 检查是否包含音频流
                has_audio = any(stream['codec_type'] == 'audio' 
                              for stream in format_info.get('streams', []))
                
                if not has_audio:
                    raise ValueError("文件中未检测到音频内容")
                
                # 转换为WAV格式
                wav_file = Path(temp_dir) / f"{uuid.uuid4()}.wav"
                cmd = [
                    'ffmpeg', '-i', str(original_file),
                    '-vn', '-acodec', 'pcm_s16le',
                    '-ac', '1', '-ar', '16000', '-y',
                    str(wav_file)
                ]
                print(f"\n执行音频转换...")
                subprocess.run(cmd, check=True, capture_output=True)
                
            except subprocess.CalledProcessError as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"音频处理失败: {e.stderr.decode()}"
                })
                return
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "无法识别的文件格式"
                })
                return
            
            # 读取转换后的 WAV 文件
            waveform, sample_rate = torchaudio.load(wav_file)
            waveform = waveform.mean(0)
            
            # 语音识别
            print("\n开始语音识别...")
            result = m.inference(
                data_in=waveform,
                language="auto",
                use_itn=True,
                output_timestamp=True,
                ban_emo_unk=True,
                fs=sample_rate,
                **kwargs
            )

            print(f"语音识别结果: {result}")
            
            if not result or not result[0]:
                raise ValueError("语音识别失败")
            
            # 4. 处理识别结果，按标点符号分段
            timestamps = result[0][0]["timestamp"]
            recognized_segments = []
            current_segment = []
            current_start = None
            
            print("\n处理识别结果...")
            for ts in timestamps:
                if len(ts) >= 3:
                    char, start_time, end_time = ts
                    
                    # 记录段落起始时间
                    if not current_start:
                        current_start = start_time
                    
                    # 遇到标点符号或最后一个字符时保存当前段落
                    if char in ['。', '，', '、', '！', '？', '.', ',', '!', '?']:
                        if current_segment:
                            segment_text = ''.join(current_segment)
                            recognized_segments.append({
                                "text": segment_text,
                                "timestamps": [current_start, end_time]
                            })
                            current_segment = []
                            current_start = None
                    else:
                        current_segment.append(char)
            
            # 处理最后一个段落
            if current_segment:
                segment_text = ''.join(current_segment)
                recognized_segments.append({
                    "text": segment_text,
                    "timestamps": [current_start, timestamps[-1][2]]
                })
            
            print(f"\n识别文本段落: {[seg['text'] for seg in recognized_segments]}")
            print(f"识别文本段落数: {len(recognized_segments)}")
            
            # 准备对齐结果
            alignment_results = []
            
            # 获取音频总时长
            total_time = recognized_segments[-1]["timestamps"][1] if recognized_segments else 0
            
            # 判断文本段落数量关系
            if len(align_segments) > len(recognized_segments):
                # 对齐文本更多，需要重新分配所有时间
                for i in range(len(align_segments)):
                    rec_text = ""
                    if i < len(recognized_segments):
                        rec_text = recognized_segments[i]["text"]
                    
                    alignment_results.append({
                        "recognizedText": rec_text,
                        "alignedText": align_segments[i],
                        "timestamps": [0, 0]  # 临时时间戳
                    })
                
                # 重新调整所有段落的时间戳
                alignment_results = adjust_timestamps(alignment_results, total_time)
                
            else:
                # 识别文本段落数量大于等于对齐文本，保持原有时间戳
                max_segments = max(len(recognized_segments), len(align_segments))
                
                for i in range(max_segments):
                    if i < len(recognized_segments):
                        rec_text = recognized_segments[i]["text"]
                        timestamps = recognized_segments[i]["timestamps"]
                    else:
                        rec_text = ""
                        last_time = recognized_segments[-1]["timestamps"][1]
                        timestamps = [last_time, last_time + 1.0]
                    
                    align_text = align_segments[i] if i < len(align_segments) else ""
                    
                    alignment_results.append({
                        "recognizedText": rec_text,
                        "alignedText": align_text,
                        "timestamps": timestamps
                    })
            
            # 返回结果
            await websocket.send_json({
                "type": "complete",
                "results": alignment_results
            })
            
    except Exception as e:
        print(f"\n处理错误: {str(e)}")
        traceback.print_exc()
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        print("\n=== 对齐请求处理完成 ===")
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
    horizontal_position: str = Form(...),
    vertical_position: str = Form(...),
    font_size: int = Form(...),
    font_color: str = Form(...),
    font_weight: str = Form(...),
    task_id: str = Form(None)
):
    try:
        print(f"\n=== 开始处理字幕合成请求 ===")
        print(f"参数信息:")
        print(f"- 水平位置: {horizontal_position}")
        print(f"- 垂直位置: {vertical_position}")
        print(f"- 字体大小: {font_size}")
        print(f"- 字体颜色: {font_color}")
        print(f"- 字体粗细: {font_weight}")
        
        # 使用绝对路径
        temp_dir = Path.cwd() / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # 保存上传的文件
        video_path = temp_dir / f"input_{uuid.uuid4()}.mp4"
        srt_path = temp_dir / f"subtitle_{uuid.uuid4()}.srt"
        output_path = temp_dir / f"output_{uuid.uuid4()}.mp4"
        
        print(f"\n保存临时文件:")
        print(f"- 视频: {video_path}")
        print(f"- 字幕: {srt_path}")
        print(f"- 输出: {output_path}")
        
        # 写入文件
        async with aiofiles.open(video_path, 'wb') as f:
            content = await video.read()
            await f.write(content)
        
        async with aiofiles.open(srt_path, 'wb') as f:
            content = await srt.read()
            await f.write(content)
        
        # 确保文件存在
        if not video_path.exists() or not srt_path.exists():
            raise Exception("临时文件创建失败")
        
        # 计算字幕位置
        vertical_align = "10" if vertical_position == "top" else "main_h-text_h-10"
        horizontal_align = {
            "left": "10",
            "center": "(main_w-text_w)/2",
            "right": "main_w-text_w-10"
        }[horizontal_position]
        
        # 计算对齐值
        def get_alignment():
            if vertical_position == 'top':
                if horizontal_position == 'left': return 7
                if horizontal_position == 'center': return 8
                return 9  # right
            else:  # bottom
                if horizontal_position == 'left': return 1
                if horizontal_position == 'center': return 2
                return 3  # right
        
        # 处理路径，确保使用正确的路径分隔符和转义
        video_path_str = str(video_path.absolute()).replace('\\', '\\\\')
        srt_path_str = str(srt_path.absolute()).replace('\\', '\\\\')
        output_path_str = str(output_path.absolute()).replace('\\', '\\\\')
        
        # 获取视频信息
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

        # 在处理视频之前，先获取视频信息
        video_info = get_video_info(video_path_str)
        width = video_info.get('width', 0)
        height = video_info.get('height', 0)

        if not width or not height:
            raise Exception("无法获取视频尺寸信息")

        # 读取 SRT 文件内容并解析时间戳和文本
        def parse_srt(srt_path):
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            subtitle_blocks = content.strip().split('\n\n')
            subtitles = []
            
            for block in subtitle_blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    # 解析时间戳
                    times = lines[1].split(' --> ')
                    start_time = times[0].replace(',', '.')
                    end_time = times[1].replace(',', '.')
                    # 获取文本（可能有多行）
                    text = ' '.join(lines[2:])
                    subtitles.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text
                    })
            
            return subtitles

        # 在 merge_subtitle 函数中替换字幕处理部分
        subtitles = parse_srt(srt_path)

        # 构建复杂的drawtext滤镜
        filter_complex = []
        for i, sub in enumerate(subtitles):
            # 转义文本中的特殊字符，使用双引号而不是单引号
            escaped_text = sub['text'].replace('"', '\\"').replace('\n', ' ')
            
            # 转换时间戳为秒数
            def timestamp_to_seconds(ts):
                h, m, s = ts.split(':')
                return float(h) * 3600 + float(m) * 60 + float(s)
            
            start_time = timestamp_to_seconds(sub['start'])
            end_time = timestamp_to_seconds(sub['end'])
            
            # 计算位置
            position = {
                'top': f"y=h*0.1",
                'bottom': f"y=h*0.9"
            }[vertical_position]
            
            if horizontal_position == 'center':
                position += ":x=(w-text_w)/2"
            elif horizontal_position == 'left':
                position += ":x=w*0.1"
            else:  # right
                position += ":x=w*0.9-text_w"
            
            # 构建单个drawtext滤镜
            filter_complex.append(
                f"drawtext=text=\"{escaped_text}\""
                f":fontsize={font_size}"
                f":fontcolor={font_color}"
                f":fontfile=/Windows/Fonts/msyh.ttc"
                f":{position}"
                f":enable='between(t,{start_time},{end_time})'"
                f":box=1:boxcolor=black@0.5:boxborderw=5"
            )

        # 构建 ffmpeg 命令，使用引号包裹滤镜字符串
        filter_string = ','.join(filter_complex)
        cmd = [
            "ffmpeg",
            "-i", video_path_str,
            "-vf", filter_string,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'copy',
            '-y',
            output_path_str
        ]

        # 打印调试信息
        print("\nDEBUG INFO:")
        print(f"Filter string: {filter_string}")
        print(f"Full Command: {' '.join(cmd)}")

        # 修改这里：使用 subprocess.run 时指定编码
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',  # 明确指定编码为 utf-8
            errors='replace',  # 处理无法解码的字符
            check=False  # 不要自动抛出异常
        )
        
        if result.returncode != 0:
            print(f"\nFFmpeg 错误输出: {result.stderr}")
            raise Exception(f"FFmpeg error: {result.stderr}")
            
        # 确保输出文件存在且大小不为0
        if not output_path.exists():
            raise Exception("输出文件生成失败")
            
        if output_path.stat().st_size == 0:
            raise Exception("输出文件大小为0")
        
        print("\n处理完成，读取输出文件")
        
        # 读取输出文件
        with open(output_path, 'rb') as f:
            video_data = f.read()
        
        return Response(
            content=video_data,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=output_{video.filename}"
            }
        )
        
    except Exception as e:
        print(f"\n处理错误: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
        
    finally:
        print("\n清理临时文件")
        # 清理临时文件
        for file in [video_path, srt_path, output_path]:
            try:
                if file.exists():
                    file.unlink()
                    print(f"- 已删除: {file}")
            except Exception as e:
                print(f"- 删除失败 {file}: {e}")
        print("=== 处理结束 ===\n")

def calculate_position(width, height, text_width, text_height, horizontal_position, vertical_position, x_offset, y_offset):
    # 水平位置计算
    if horizontal_position == "center":
        x = (width - text_width) // 2
    elif horizontal_position == "right":
        x = width - text_width - int(width * 0.1)  # 10% 边距
    else:  # left
        x = int(width * 0.1)  # 10% 边距

    # 垂直位置计算
    if vertical_position == "bottom":
        y = height - text_height - int(height * 0.1)  # 10% 边距
    elif vertical_position == "middle":
        y = (height - text_height) // 2
    else:  # top
        y = int(height * 0.1)  # 10% 边距

    # 应用偏移
    x += x_offset
    y += y_offset
    
    return x, y

@app.post("/api/v1/add-watermark")
async def add_watermark(
    video: UploadFile,
    watermark_text: Annotated[str, Form()],
    watermark_size: Annotated[int, Form()] = 24,
    watermark_color: Annotated[str, Form()] = "FFFFFF",
    watermark_opacity: Annotated[int, Form()] = 50,
    watermark_position: Annotated[str, Form()] = "bottom-right",
    padding: Annotated[int, Form()] = 20,
    task_id: Annotated[str, Form()] = None
):
    try:
        # 创建临时文件
        video_path = Path(tempfile.gettempdir()) / f"input_{uuid.uuid4()}.mp4"
        output_path = Path(tempfile.gettempdir()) / f"output_{uuid.uuid4()}.mp4"
        final_output = Path(tempfile.gettempdir()) / f"final_{uuid.uuid4()}.mp4"

        # 保存上传的视频
        with open(video_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)

        # 获取视频信息
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # 加载字体
        font_path = "C:\\Windows\\Fonts\\msyh.ttc"
        if not os.path.exists(font_path):
            font_path = "C:\\Windows\\Fonts\\arial.ttf"
        font = ImageFont.truetype(font_path, watermark_size)

        # 处理颜色和透明度
        watermark_rgb = tuple(int(watermark_color[i:i+2], 16) for i in (0, 2, 4))
        watermark_alpha = int(watermark_opacity * 255 / 100)
        watermark_rgba = (*watermark_rgb, watermark_alpha)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为PIL图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)

            # 计算水印文本大小
            text_bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 计算水印位置
            if watermark_position == "top-left":
                x, y = padding, padding
            elif watermark_position == "top-right":
                x = width - text_width - padding
                y = padding
            elif watermark_position == "bottom-left":
                x = padding
                y = height - text_height - padding
            else:  # bottom-right
                x = width - text_width - padding
                y = height - text_height - padding

            # 绘制水印
            draw.text((x, y), watermark_text, font=font, fill=watermark_rgba)

            # 转回OpenCV格式
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            out.write(frame)

            # 更新进度
            frame_count += 1
            if task_id in progress_connections:
                try:
                    await progress_connections[task_id].send_json({
                        "progress": int(frame_count * 100 / total_frames),
                        "frame": frame_count,
                        "total": total_frames
                    })
                except Exception as e:
                    print(f"Error sending progress: {e}")

            # 每处理一定数量的帧后让出控制权
            if frame_count % 10 == 0:
                await asyncio.sleep(0)

        # 释放资源
        cap.release()
        out.release()

        # 使用ffmpeg重新编码
        cmd = [
            'ffmpeg',
            '-i', str(output_path),
            '-i', str(video_path),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
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
                "Content-Disposition": f"attachment; filename=watermark_{video.filename}"
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:
        # 清理临时文件
        for file in [video_path, output_path, final_output]:
            try:
                if file.exists():
                    file.unlink()
            except Exception as e:
                print(f"Error deleting {file}: {e}")


 # 使用模型进行音频解析
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    ban_emo_unk=True,
)


@app.post("/api/v1/parse-audio")
async def parse_audio(file: UploadFile):
    """
    解析音频文件接口
    - file: 音频文件(mp3)
    """
    try:
        # 读取音频文件
        content = await file.read()
        
        # 获取文件扩展名
        file_extension = os.path.splitext(file.filename)[1]
        if not file_extension:
            file_extension = '.mp4'  # 默认扩展名
            
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(content)
            original_path = temp_file.name

        # 创建截取后的临时文件路径
        trimmed_path = original_path.replace(file_extension, f'_trimmed{file_extension}')
        
        # 使用 ffmpeg 截取并处理音频
        out, _ = (
            ffmpeg
            .input(original_path)
            .filter('atrim', duration=20)  # 截取前30秒
            .output(trimmed_path,
                    ar='16000',  # 采样率
                    ac='1',      # 单声道
                    format='wav',
                    acodec='pcm_s16le',  # 使用16位PCM编码
                    audio_bitrate='64k',  # 降低比特率
                    compression_level='5'  # 压缩级别
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        # 使用处理后的文件调用模型
        res = model.generate(
            input=trimmed_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
            ban_emo_unk=True,
        )
        
        # 处理返回结果
        text = rich_transcription_postprocess(res[0]["text"])
        cleaned_text = clean_text(text)
        
        return {
            "success": True,
            "text": cleaned_text,
            "result": res
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
    finally:
        # 清理临时文件
        try:
            if os.path.exists(original_path):
                os.unlink(original_path)
            if os.path.exists(trimmed_path):
                os.unlink(trimmed_path)
        except Exception as e:
            print(f"Error deleting temporary files: {e}")

def clean_text(text: str) -> str:
    """
    清理文本，去除多余符号和表情
    """
    # 使用正则表达式去除所有表情符号，保留中文和常见标点符号
    cleaned = re.sub(r'[^\w\s,.!?，。！？\u4e00-\u9fa5]', '', text)  # 保留中文字符
    return cleaned.strip()  # 去除首尾空格

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
