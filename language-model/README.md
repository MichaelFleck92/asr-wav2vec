## Add a Language Model to Wav2Vec2

This code snippets were used to train a n-gram model, which got added to the fine-tuned model. It uses a corpus of translations of the europarliament and the Common-Voice data itself.
It follows the tutorial avaialable [here](https://huggingface.co/blog/wav2vec2-with-ngram)

Build a text file containing raw training data:
```bash
python3 lm1.py
```
  
Build n-gram model with kenlm:
```bash
kenlm/build/bin/lmplz -o 5 <"text.txt" > "5gram.arpa"
```
  
Add n-gram model to Wav2Vec2:
```bash
python3 lm2.py
```

For needed dependencies and more information visit the linked tutorial.
