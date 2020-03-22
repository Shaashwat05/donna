from keras.models import Model,Input

def inference_model():
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h=Input(shape=(200, ))
    decoder_state_input_c=Input(shape=(200, ))    

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c =decoder_lstm(decoder_embedding, initial_state = decoder_states_inputs)
    decoder_states =[state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( tokenizer.word_index[ word ] ) 
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')