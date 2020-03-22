from simple_chatbot import preprocess, train
from inf_model import inference_model,str_to_tokens
from keras.models import load_model
import numpy as np


no_questions, no_answers, encoder_input_data, decoder_input_data, decoder_target_data=preprocess()
model.train()

model=load_model("simlpe_model.h5")
encoder_model, decoder_model = inference_model()


for i in range(10):
    states_values = encoder_model.predict(str_to_tokens(input("enter a question")))
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0,0] = to

