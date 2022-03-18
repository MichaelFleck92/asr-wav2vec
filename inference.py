from transformers import pipeline

pipe = pipeline(model="mfleck/wav2vec2-large-xls-r-300m-german-with-lm")
output = pipe("/path/to/file.wav",chunk_length_s=5, stride_length_s=1)
print(output["text"])