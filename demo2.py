#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess


model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
m.eval()

# res = m.inference(
#     data_in=f"E:/project/SenseVoice/temp_segments/69bca089-55ef-48f0-b6ec-7f59a046ae46/segment_0.mp3",
#     language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=False,
#     ban_emo_unk=False,
#     **kwargs,
# )

# text = rich_transcription_postprocess(res[0][0]["text"])
# print(text)

res = m.inference(
    data_in=f"E:/project/SenseVoice/temp_segments/69bca089-55ef-48f0-b6ec-7f59a046ae46/segment_0.mp3",
    language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    ban_emo_unk=True,
    output_timestamp=True,
    **kwargs,
)

timestamp = res[0][0]["timestamp"]
text = rich_transcription_postprocess(res[0][0]["text"])
print('text:',text)
print('timestamp:',timestamp)
