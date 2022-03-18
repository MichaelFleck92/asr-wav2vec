import argparse
import re
from typing import Dict

import torch
from datasets import Audio, Dataset, load_dataset, load_metric

from transformers import AutoFeatureExtractor, pipeline



# load dataset
dataset = load_dataset("common_voice", "de", split="test")
# use only 1% of data
#dataset = load_dataset("common_voice", "de", split="test[:1%]")


# load processor
feature_extractor = AutoFeatureExtractor.from_pretrained("mfleck/wav2vec2-large-xls-r-300m-german-with-lm")
sampling_rate = feature_extractor.sampling_rate

dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

# load eval pipeline
# device=0 means GPU, use device=-1 for CPU
asr = pipeline("automatic-speech-recognition", model="mfleck/wav2vec2-large-xls-r-300m-german-with-lm", device=0)

# Remove batches with chars which do not exist in German
regex = "[^A-Za-zäöüÄÖÜß,?.! ]+"
dataset = dataset.filter(lambda example: bool(re.search(regex, example['sentence']))==False)

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
# map function to decode audio
def map_to_pred(batch):
    prediction = asr(batch["audio"]["array"], chunk_length_s=5, stride_length_s=1)

    # Print automatic generated transcript
    #print(str(prediction))

    batch["prediction"] = prediction["text"]
    text = batch["sentence"]
    batch["target"] = re.sub(chars_to_ignore_regex, "", text.lower()) + " "
    
    return batch

# run inference on all examples
result = dataset.map(map_to_pred, remove_columns=dataset.column_names)

# load metric
wer = load_metric("wer")
cer = load_metric("cer")

# compute metrics
wer_result = wer.compute(references=result["target"], predictions=result["prediction"])
cer_result = cer.compute(references=result["target"], predictions=result["prediction"])

# print results
result_str = f"WER: {wer_result}\n" f"CER: {cer_result}"
print(result_str)





