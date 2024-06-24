import torch
import random
def word_to_one_hot(word, word_to_index):
    one_hot = [0] * len(word_to_index)
    index = word_to_index[word]
    one_hot[index] = 1
    return one_hot

def count_max_sentence_length(data):
    max_word_count = 0
    for sentence in data:
        current_count = len(sentence)
        max_word_count = max_word_count if max_word_count > current_count else current_count
    return max_word_count

def sentence_padding(data):
    max_word_count = count_max_sentence_length(data)
    pad_element = ['<PAD>']
    for sentence in data:
        current_count = len(sentence)
        if current_count < max_word_count:
            sentence += pad_element * (max_word_count - current_count)
    return max_word_count, data

def data_loader(word_to_one_hot, batch_size):
    bucket = random.sample(word_to_one_hot, len(word_to_one_hot))
    bucket = [bucket[i : i + batch_size] for i in range(0, len(bucket), batch_size)]
    random.shuffle(bucket)
    for batch in bucket:
        batch = torch.tensor(batch, dtype = torch.float64)
        yield batch