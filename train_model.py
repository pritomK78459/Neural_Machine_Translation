from model_creation import CreateModel
from data_preperation import PrepareData
import numpy as np
import tensorflow

class TrainModel(CreateModel):
    def __init__(self, batch_size=64,epochs=10):
        CreateModel.__init__(self)
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, encoder_input_data, decoder_input_data, decoder_target_data, outdir):
        print("Training the model ....")
        model, encoder_model, decoder_model = self.model_enc_dec()
        model.compile(optimizer='adam',loss='categorical_crossentropy')
        model.fit(x=[encoder_input_data,decoder_input_data],y=decoder_target_data,
                  batch_size=self.batch_size,epochs=self.epochs,validation_split=0.2)

        model.save(outdir + 'eng_2_fre.h5')
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
        print(self.target_word_index)
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
    input_text_preprocess, target_text_preprocess = train_model.vocab_generation('./fra.txt', 2000)
    encoder_input_data, decoder_input_data, decoder_target_data, input_texts, target_texts = train_model.process_inputs(input_text_preprocess, target_text_preprocess)
    create_model = CreateModel(1000)
    train_indices, test_indices = train_model.train_test_split(2000)
    train_encoder_input_data = encoder_input_data[train_indices]
    test_encoder_input_data = encoder_input_data[test_indices]
    train_decoder_input_data = decoder_input_data[train_indices]
    test_decoder_input_data = decoder_input_data[test_indices]
    train_decoder_target_data = decoder_target_data[train_indices]
    test_decoder_target_data = decoder_target_data[test_indices]
    train_input_texts = input_texts[train_indices]
    test_input_texts = input_texts[test_indices]
    model, encoder_model1, decoder_model1 = train_model.train(train_encoder_input_data, train_decoder_input_data, train_decoder_target_data,'./models/')
    #encoder_input_data_inference, _, _, input_texts_inference, _ = train_model.process_inputs(input_text_preprocess,target_text_preprocess)
    in_list, out_list = train_model.inference(model, test_encoder_input_data, encoder_model1, decoder_model1, test_input_texts)


