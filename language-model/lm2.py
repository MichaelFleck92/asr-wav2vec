from transformers import AutoProcessor
from pyctcdecode import build_ctcdecoder
from huggingface_hub import Repository


with open("5gram.arpa", "r") as read_file, open("5gram_correct.arpa", "w") as write_file:
    has_added_eos = False
    for line in read_file:
    if not has_added_eos and "ngram 1=" in line:
        count=line.strip().split("=")[-1]
        write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
    elif not has_added_eos and "<s>" in line:
        write_file.write(line)
        write_file.write(line.replace("<s>", "</s>"))
        has_added_eos = True
    else:
        write_file.write(line)


processor = AutoProcessor.from_pretrained("mfleck/wav2vec2-large-xls-r-300m-german-with-lm")

vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}


decoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path="/home/debian/boost/5gram_correct.arpa",
)

from transformers import Wav2Vec2ProcessorWithLM

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)



repo = Repository(local_dir="wav2vec2-large-xls-r-300m-german-with-lm", clone_from="mfleck/wav2vec2-large-xls-r-300m-german-with-lm")

processor_with_lm.save_pretrained("wav2vec2-large-xls-r-300m-german-with-lm")