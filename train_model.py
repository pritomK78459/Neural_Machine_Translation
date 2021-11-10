from model_creation import CreateModel
from data_preperation import PrepareData
import numpy as np
import tensorflow

class TrainModel(CreateModel):
    def __init__(self, batch_size=64,epochs=10):
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, encoder_input_data, decoder_input_data,decoder_target_data):
        print("Training the model ....")
        model, encoder_model, decoder_model = self.model_enc_dec()
        model.compile(optimizer='adam',loss='categorical_crossentropy')
        model.fit(x=[encoder_input_data,decoder_input_data],y=decoder_target_data,
                  batch_size=self.batch_size,epochs=self.epochs,validation_split=0.2)

        model.save(self.outdir + 'eng_2_fre.h5')
        return model, encoder_model, decoder_model

    def train_test_split(self, num_recs, train_frac=0.8):
        rec_indices = np.arange(num_recs)
        np.random.shuffle(rec_indices)
        train_count = int(num_recs * 0.8)
        train_indices = rec_indices[:train_count]
        test_indices = rec_indices[train_count:]

        return train_indices, test_indices

    def decode_sequence(self, input_sequence, encoder_model, decoder_model):
        states_value = encoder_model.predict(input_sequence)
        target_sequence = np.zeros((1,1, self.num_decoder_words))
        target_sequence[0, 0, self.target_word_index['\t']] = 1.

        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            output_word, h, c = decoder_model.predict(
                [target_sequence] + states_value
            )
            sampled_word_index = np.argmax(output_word[0, -1, :])
            sampled_char = self.reverse_target_word_dict[sampled_word_index]
            decoded_sentence = decoded_sentence + ' ' + sampled_char
            if sampled_char == '\n' or len(decoded_sentence) > self.max_decoder_seq_length:
                stop_condition = True

            target_sequence = np.zeros((1, 1, self.num_decoder_words))
            target_sequence[0, 0, sampled_word_index] = 1.
            states_value = [h, c]

        return decoded_sentence

    def inference(self, model, data, encoder_model, decoder_model, in_text):
        in_list, out_list = [],[]
        for seq_index in range(data.shape[0]):
            input_seq = data[seq_index: seq_index + 1]
            decoded_sentence = self.decode_sequence(input_seq, encoder_model, decoder_model)
            print('-')
            print('Input sentence: ', in_text[seq_index])
            print('Decoded sentence: ', decoded_sentence)
            in_list.append(in_text[seq_index])
            out_list.append(decoded_sentence)
        return in_list, out_list

if __name__ == '__main__':
    train_model = TrainModel()
    prepare_data = PrepareData('train')
    input_texts, target_texts = prepare_data.vocab_generation('./fra.txt', 20000)
    encoder_input_data, decoder_input_data, decoder_target_data, input_texts, target_texts = prepare_data.process_inputs(input_texts,target_texts)
    create_model = CreateModel(7)
    model, encoder_model, decoder_model = create_model.model_enc_dec()

