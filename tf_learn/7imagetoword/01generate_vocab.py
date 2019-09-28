import pprint

input_description_file = './datas/results_20130124.token'
output_vocab_file = './datas/vocab.txt'

def count_vocab(input_description_file):
    """Generates vocabulary.
    In addition, count distribution of length of sentence
    and max length of image description.
    """
    with open(input_description_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    max_length_of_sentences = 0
    length_dict = { }
    vocab_dict = { }
    for line in lines:
        image_id, description = line.strip('\n').split('\t')
        words = description.strip(' ').split()
        max_length_of_sentences = max(max_length_of_sentences, len(words))
        length_dict.setdefault(len(words),0)
        length_dict[len(words)] += 1

        for word in words:
            vocab_dict.setdefault(word, 0)
            vocab_dict[word] += 1

    print(max_length_of_sentences)
    pprint.pprint(length_dict)
    return vocab_dict

vocab_dict = count_vocab(input_description_file)
print(vocab_dict.items())
sorted_vocab_dict = sorted(vocab_dict.items(), key=lambda d:d[1], reverse=True)
with open(output_vocab_file, 'w', encoding='utf-8') as f:
    f.write('<UNK>\t1000000\n')
    for item in sorted_vocab_dict:
        f.write('%s\t%d\n' % (item[0],item[1]))