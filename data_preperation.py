import numpy as np

class PrepareData:

    def __init__(self,mode='train'):
        self.mode = mode

    def read_files(self, path, num_samples):
        input_texts = []
        target_texts = []
        input_words = set()
        target_words = set()

        with open(path, 'r', encoding='utf-8') as file:
            lines = file.read().split('\n')
        for line in lines[: min(num_samples, len(lines) -1)]:
            input_text, target_text = line.split("\t")[:-1]
            target_text = '\t ' + target_text + ' \n'
            input_texts.append(input_text)
            target_texts.append(target_text)

            for word in input_text.split(" "):
                if word not in input_words:
                    input_words.add(word)
            for word in target_text.split(" "):
                if word not in target_words:
                    target_words.add(word)

        return input_texts, target_texts, input_words, target_words

    def vocab_generation(self, path, num_samples):

        input_texts, target_texts, input_words, target_words = self.read_files(path, num_samples)
        input_words = sorted(list(input_words))
        target_words = sorted(list(target_words))
        self.num_encoder_words = len(input_words)
        self.num_decoder_words = len(target_words)
        self.max_encoder_seq_length = max([len(text.split(" ")) for text in input_texts])
        self.max_decoder_seq_length = max([len(text.split(" ")) for text in target_texts])

        self.input_word_index = dict([(word ,i) for i, word in enumerate(input_words)])
        self.target_word_index = dict([(word ,i) for i, word in enumerate(target_words)])
        self.reverse_input_word_dict = dict((i,word) for word, i in self.input_word_index.items())
        self.reverse_target_word_dict = dict((i,word) for word, i in self.target_word_index.items())

        return input_texts, target_texts

    def process_inputs(self, input_texts, target_texts=None):
        encoder_input_data = np.zeros((len(input_texts), self.max_encoder_seq_length, self.num_encoder_words), dtype='float32')
        decoder_input_data = np.zeros((len(input_texts), self.max_decoder_seq_length, self.num_decoder_words), dtype='float32')
        decoder_target_data = np.zeros((len(input_texts), self.max_decoder_seq_length, self.num_decoder_words), dtype='float32')

        if self.mode == 'train':
            for i,(input_text, target_text) in enumerate(zip(input_texts, target_texts)):
                for t, word in enumerate(input_text.split(" ")):
                    try:
                        encoder_input_data[i, t, self.input_word_index[word]] = 1.
                    except:
                        print(f'word {word} encountered for the 1st time, skipped')
                for t, word in enumerate(target_text.split(" ")):
                    decoder_input_data[i, t, self.target_word_index[word]] = 1.
                    if t > 0:
                        try:
                            decoder_target_data[i,t-1, self.target_word_index[word]] = 1.
                        except:
                            print(f'word {word} encountered for the 1st time, skipped')
            return encoder_input_data, decoder_input_data, decoder_target_data, np.array(input_texts), np.array(target_texts)
        else:
            for i, input_text in enumerate(input_texts):
                for t, word in enumerate(input_text.split(" ")):
                    try:
                        encoder_input_data[i, t, self.input_word_index[word]] = 1.
                    except:
                        print(f'word {word} encountered for the 1st time, skipped')

            return encoder_input_data, None, None, np.array(input_texts), None



