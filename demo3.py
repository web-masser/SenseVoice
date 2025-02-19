from funasr import AutoModel

model = AutoModel(model="fsmn-vad")

wav_file = f"E:/M800002mRbOA2rFHEV.mp3"
res = model.generate(input=wav_file)
print(res)


