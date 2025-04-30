from tokenizer import KazakhLMTokenizer


kz_tokenizer = KazakhLMTokenizer('kaz_news_2020_30K-sentences.txt', 500)

kz_tokenizer.train_tokenizer()

vocab = kz_tokenizer.vocab

for key, val in vocab.items():
    print(key, val.decode('utf-8', errors='replace'))
    
print(f"Now your vocab length = {len(vocab)}")