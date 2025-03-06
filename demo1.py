#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    ban_emo_unk=True,
)

# data_in=f"E:/project/SenseVoice/temp_segments/69bca089-55ef-48f0-b6ec-7f59a046ae46/segment_0.mp3",

# en
res = model.generate(
    input=f"./temp/a650c708-12b4-4a04-b1f3-79a3d3535896.wav",
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
    ban_emo_unk=True,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text, 'text')
print(res, 'res')

