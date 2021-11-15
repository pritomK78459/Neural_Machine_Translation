from data_preperation import PrepareData
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow as tf

class CreateModel(PrepareData):
    def __init__(self, latent_dim=2000):
        PrepareData.__init__(self)
        self.latent_dim = latent_dim

    def model_enc_dec(self):
        encoder_input = Input(shape=(None, self.num_encoder_words), name='encoder_input')
        encoder = LSTM(self.latent_dim, return_state=True, name='encoder')
        encoder_out, state_h, state_c = encoder(encoder_input)
        encoder_states = [state_h, state_c]

        decoder_input = Input(shape=(None, self.num_decoder_words), name='decoder_input')
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_out, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_words, activation='softmax', name='decoder_dense')
        decoder_out = decoder_dense(decoder_out)

        model = Model(inputs=[encoder_input, decoder_input],outputs=decoder_out)
        encoder_model = Model(inputs=encoder_input, outputs=encoder_states)

        decoder_inp_h = Input(shape=(self.latent_dim,))
        decoder_inp_c = Input(shape=(self.latent_dim,))
        decoder_input_new = Input(shape=(None, self.num_decoder_words,))
        decoder_inp_state = [decoder_inp_h,decoder_inp_c]
        decoder_out, decoder_out_h, decoder_out_c = decoder_lstm(inputs = decoder_input_new, initial_state=decoder_inp_state)
        decoder_out = decoder_dense(decoder_out)
        decoder_out_state = [decoder_out_h, decoder_out_c]
        decoder_model = Model(inputs=[decoder_input_new] + decoder_inp_state, outputs=[decoder_out] + decoder_out_state)

        return model, encoder_model, decoder_model


