from datasets import load_dataset
import re
from huggingface_hub import notebook_login

dataset = load_dataset("europarl_bilingual", lang1="de", lang2="en", split="train")

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\‘\”\�\'\(\)]'

print(dataset[10])
def extract_text(batch):
	text = batch["translation"]["de"]
	batch["text"] = re.sub(chars_to_ignore_regex, "", text.lower())
	return batch

dataset = dataset.map(extract_text, remove_columns=dataset.column_names)

with open("text.txt", "w") as file:
	file.write(" ".join(dataset["text"]))



dataset = load_dataset('common_voice', 'de', split="train+validation+test")
dataset = dataset.remove_columns(["audio", "accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

print(dataset[10])

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\‘\”\�\'\(\)]'
def extract_text(batch):
	batch["text"] = re.sub(chars_to_ignore_regex, "", batch["sentence"].lower())
	#print(batch["text"])
	return batch

dataset = dataset.map(extract_text)

# Remove batches with chars which do not exist in German
print(len(dataset))
regex = "[^A-Za-zäöüÄÖÜß ]+"
dataset = dataset.filter(lambda example: bool(re.search(regex, example['text']))==False)
print(len(dataset))

print(dataset[10])

with open("text.txt", "a") as file:
	file.write(" ".join(dataset["text"]))





dataset.push_to_hub(f"german_extracted_text", split="train")