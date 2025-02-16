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
        # 读取音频文件
        content = await file.read()
        audio_io = BytesIO(content)
        waveform, sample_rate = torchaudio.load(audio_io)
        waveform = waveform.mean(0)

        # TODO: 实现文本对齐逻辑
        # 目前模型代码中没有直接的文本对齐功能
        # 需要进一步了解具体的对齐实现方式

        return {
            "success": True,
            "message": "Text alignment feature is under development"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

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
