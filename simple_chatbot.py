import os
import yaml
import keras
from keras import preprocessing,utils
from keras.layers import Input,Embedding,Dense,LSTM
from keras.models import Model
from keras.optimizers import RMSprop
from keras.activations import softmax
import numpy as np


def preprocess():
    dir_path="/home/shaashwatlobnikki/Desktop/donna/simple_dataset"
    files_list=os.listdir(dir_path+os.sep)

    questions=[]
    answers=[]

    for filepath in files_list:
        stream = open(dir_path+os.sep+filepath, 'rb')
        docs=yaml.safe_load(stream)
        conversations = docs['conversations']
        for con in conversations:
            if(len(con)>2):
                questions.append(con[0])
                replies=con[1:]
                ans=''
                for rep in replies:
                    ans+=' '+rep
                answers.append(ans)
            elif(len(con)>1):
                questions.append(con[0])
                answers.append(con[1])

    print(len(answers))

    ####preprocessing questions/encoder input
    tokenizer1=preprocessing.text.Tokenizer()
    tokenizer1.fit_on_texts(questions)
    tokenized_questions=tokenizer1.texts_to_sequences(questions)

    length_list1=[]
    for token_seq in tokenized_questions:
        length_list1.append(len(token_seq))
    max_input_length=np.array(length_list1).max()

    padded_questions=preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=max_input_length , padding='post' )
    encoder_input_data = np.array( padded_questions )

    questions_dict=tokenizer1.word_index
    no_questions=len(questions_dict)+1


    ####preprocessing answers/decoder input
    for line in range(len(answers)):
        answers[line]='<START>' + str(answers[line]) + '<END>'

    tokenizer2 = preprocessing.text.Tokenizer()
    tokenizer2.fit_on_texts(answers) 
    tokenized_answers = tokenizer2.texts_to_sequences(answers) 

    length_list2= []
    for token_seq in tokenized_answers:
        length_list2.append( len( token_seq ))
    max_output_length = np.array( length_list2 ).max()

    padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=max_output_length, padding='post' )
    decoder_input_data = np.array( padded_answers )

    answers_dict = tokenizer2.word_index
    num_answers = len( answers_dict )+1


    ####preprocessing answers/decoder target data
    decoder_target_data=[]
    for token_seq in tokenized_answers:
        decoder_target_data.append(token_seq[1:])

    padded_answers2=preprocessing.sequence.pad_sequences(decoder_target_data , maxlen=max_output_length, padding='post' )

    onehot_answers2=utils.to_categorical(padded_answers2 , num_answers)
    decoder_target_data = np.array( onehot_answers2 )

    return no_questions, no_answers, encoder_input_data, decoder_input_data, decoder_target_data



def train():
    ####seq-to-seq model
    encoder_inputs=Input(shape=(None,))
    encoder_embedding=Embedding(no_questions,256,mask_zero=True)(encoder_inputs)
    encoder_outputs,state_h,state_c=LSTM(128,return_state=True)(encoder_embedding)
    encoder_states=[state_h,state_c]

    decoder_inputs=Input(shape=(None,))
    decoder_embedding=Embedding(num_answers,256,mask_zero=True)(decoder_inputs)
    decoder_lastm=LSTM(128,return_state=True,return_sequences=True)
    decoder_outputs,_,_=decoder_lastm(decoder_embedding, initial_state=encoder_states)
    decoder_dense=Dense(num_answers,activation=softmax)
    output=decoder_dense(decoder_outputs)

    model=Model([encoder_inputs,decoder_inputs], output)
    model.compile(optimizer=RMSprop(),loss='categorical_crossentropy')
    model.summary()

    #print(encoder_input_data.shape)
    #print(decoder_input_data.shape)
    #print(decoder_target_data.shape)

    model.fit([encoder_input_data , decoder_input_data], decoder_target_data, batch_size=250, epochs=50) 
    model.save( 'simple_model.h5' ) 
