from tokenizer import KazakhLMTokenizer


kz_tokenizer = KazakhLMTokenizer('kaz_news_2020_30K-sentences.txt', 1000)

text = "Сәлем досым, қалайсың?"
encoded_text_before_train = kz_tokenizer.encode(text)

kz_tokenizer.train_tokenizer()

vocab = kz_tokenizer.vocab

for key, val in vocab.items():
    print(key, val.decode('utf-8', errors='replace'))
    
print(f"Now your vocab length = {len(vocab)}")

encoded_text_after_train = kz_tokenizer.encode(text)

print(f'Original text: {text}')
print(f'Encoded text before train: {encoded_text_before_train}. And its length: {len(encoded_text_before_train)}')
print(f'Encoded text: {encoded_text_after_train}. And its length: {len(encoded_text_after_train)}')

kz_tokenizer.save_merges('merges_test.pkl') 
kz_tokenizer.save_vocab('vocab_test.pkl') 