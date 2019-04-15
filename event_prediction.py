''' 
		
		In this script, you'll find all important functions for the next event prediction :
	- feature ectractor functions "get_array_file" that exctrat a feature vector out of a 15 min xml file
	- "build_model_team_prediction" that build and load weights of the next team prediction
	- 

		
'''



###################################################################################
#################################### Packages #####################################
###################################################################################

from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Embedding, Dense,  \
                Concatenate, Activation, Flatten, Input, BatchNormalization, Dropout, concatenate
from keras.optimizers import Adam
from keras.models import Model

from lxml import etree, objectify

import pandas as pd
import numpy as np
import os
from tqdm import tqdm, tqdm_notebook

import warnings
warnings.filterwarnings('ignore')


from collections import Counter
from player_prediction import event_to_idx, q_to_idx
from load_data import parse_xml_file, event_df_columns, game_df_columns, q_df_columns

###################################################################################
########################### Functions - Team Prediction ###########################
###################################################################################

def build_model_team_prediction(weights_path = 'team_prediction.h5'):

    ''' Build and load weights of the model that predicts the team of the next event.
    It is a three tower neural network that uses knowledge on the event_id to build 
    a small embedding, on a feature vector of the previous 15 min and finally on the 
    next 10 events.'''

    coord_in = Input(shape=(10,3,),name='input_2')       
    coord = Model(inputs=coord_in, outputs=coord_in)

    # Event ID Encoder 
    caption_in = Input(shape=(10, ), name='input_1')
    y = Embedding(77, 10, input_length=10, name ='embedding_1')(caption_in)
    type_event = Model(inputs=caption_in, outputs=y)

    vec_in = Input(shape=(200, ), name='input_3')
    vec_model = Model(inputs=vec_in, outputs=vec_in)

    merged = concatenate([type_event.output, coord.output, ], axis=2, name ='merge_1')
    merged = LSTM(20)(merged)
    merged = concatenate([merged, vec_model.output ], axis=1, name ='merge_2')
    merged = Dense(5, activation='sigmoid')(merged)
    merged = Dense(1, activation='sigmoid')(merged)

    final_model = Model([type_event.input,coord.input, vec_model.output], merged)
    final_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Load weigths. By default 'team_prediction.h5'
    final_model.load_weights(weights_path)

    return final_model

def get_array_file(file):

    ''' Construct the feature vector out of a 15 min xml file, that will be feed into 
    the neural networks for the prediction of the next team.

    This feature vectors basically consists in the aggregation for both team of all different 
    event occuring (related to one team at a time) and the pourcentage of positive outcome for 
    each event 

    Alongside this vector, we return as well the a (10,3) array that gather information on the position 
    (x,y), the id of the team as well as the type_id of the past 10 events.) '''
    
    batch_input = []
    batch_vec = []

    game_df, event_df, q_df = parse_xml_file(file)
    
    home_team_id, away_team_id = game_df['home_team_id'][0], game_df['away_team_id'][0]
    #print('ID',home_team_id, away_team_id)
    #event_df['team_id'] = event_df['team_id'].map({home_team_id: '1', away_team_id:'0'})
    event_df['min']= event_df['min'].astype(int)
    event_df['x']= event_df['x'].astype(float)
    event_df['y']= event_df['y'].astype(float)
    event_df['period_id']= event_df['period_id'].astype(int)

    maxx = event_df['x'].max()
    maxy = event_df['y'].max()

    event_df = event_df[event_df['min'].between(1,90)]


    # Select the 300 events before the 10 anonymised event
    past_df = event_df.loc[event_df.index[:-10]]
    past_df['outcome'] = past_df['outcome'].astype(int)

    # Select the 10 last ananumised event
    df_chunk = event_df.loc[event_df.index[-10:]]

    # We're gonna collect the two vectors of the past events aggregated by team 
    vec = []
    for team in range(2) :
        # Select the concerned team
        df_event_chunk = past_df[past_df['team_id']==str(team)]
        #print(past_df.shape,df_event_chunk.shape, past_df.team_id.unique())
        #Get occurances of each event type
        dic_event = dict(Counter(df_event_chunk.type_id))
        #print(dic_event)
        # List of keys
        list_keys_event = list(dic_event.keys())
        vec_e = np.zeros(50*2)

        for feature in list_keys_event:
            mapped = event_to_idx.get(feature,None)
            if mapped != None:
                l = list(df_event_chunk[df_event_chunk['type_id']==feature].outcome)

                if l != []:
                    mean = np.round(np.mean(l),3)
                    vec_e[mapped*2+1] = mean
                    #print(mean)
                vec_e[mapped*2] = dic_event[feature]
            else : 
                pass
        vec.append(vec_e)
    # Vec of event with outcome ratio
    vec = np.concatenate(vec, axis=0)

    batch_vec.append(vec[np.newaxis,:])

    df_chunk = df_chunk[['team_id','x', 'y','type_id']]
    batch_input_vec = np.array(df_chunk)
    batch_input_vec[:,1] /= maxx
    batch_input_vec[:,2] /= maxy
    batch_input.append(batch_input_vec[np.newaxis,:,:])
        
    return np.concatenate(batch_vec, axis=0), np.concatenate(batch_input, axis=0)


###################################################################################
########################## Functions - (X,Y) Prediction ###########################
###################################################################################


from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import LSTM, Dropout

def load_model_x_y(weights_path = 'x_y.h5'):

    ''' Build and Load the model that predicts the next position (x,y) '''
    x = Input(shape=(10,53,),name='input_2')       
    y = BatchNormalization(name='batch_normalization_1')(x)
    y = LSTM(10)(y)
    y = Dense(2, activation='relu')(y)

    model = Model(inputs=x, outputs=y)
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    
    # Load weights. By default 'x_y.h5'
    model.load_weights(weights_path)

    return model

def get_array_file_position(xml_file):
    
    game_df, event_df, q_df = parse_xml_file(xml_file)
    event_df['min']= event_df['min'].astype(int)
    event_df['x']= event_df['x'].astype(float)
    event_df['y']= event_df['y'].astype(float)
    event_df['team_id']= event_df['team_id'].astype(int)
    #maxx = event_df['x'].max()
    #maxy = event_df['y'].max()
    
    df_chunk = event_df.loc[event_df.index[-10:]]
    # Get output
    df_chunk = df_chunk[['team_id','x', 'y','type_id']]
    
    batch_input = np.zeros((10,53))
    for i,val in enumerate(df_chunk.type_id.values) :
        batch_input[i,event_to_idx[val]] = 1
    batch_input[:,-3] = df_chunk.x.values
    batch_input[:,-2] = df_chunk.y.values
    batch_input[:,-1] = df_chunk.team_id.values

    return batch_input[np.newaxis,:,:]