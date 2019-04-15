''' In the scipt : 
    - we load some pickle files and defines some mapping dictionnnaries.
    - we define two feature extractor functions
    - and we define the player prediction model 
'''

###################################################################################
#################################### Packages #####################################
###################################################################################

from load_data import parse_xml_file, event_df_columns, game_df_columns, q_df_columns
from team_trick import filtered_indices_team, active_player_800, player_to_idx, idx_to_player, dictionnary_time_per_player, dic_team, dic_team_active, team_idx
from collections import Counter
import joblib

import pandas as pd
import numpy as np
import pickle
    
###################################################################################
######################## Pickle and Dictionnaries #################################
###################################################################################

with open('qualifier_ids.pkl', 'rb') as handle:
    qualifier_ids = pickle.load(handle)
    
with open('event_ids.pkl', 'rb') as handle:
    event_ids = pickle.load(handle)
    
q_to_idx = {}
for i,j in enumerate(qualifier_ids):
    q_to_idx[str(j)] = i
    

event_to_idx = {}
for i,j in enumerate(event_ids):
    event_to_idx[str(j)] = i
    
idx_to_q = {}
for i,j in enumerate(qualifier_ids):
    idx_to_q[i] = int(j)
    
idx_to_event = {}
for i,j in enumerate(event_ids):
    idx_to_event[i] = int(j)

###################################################################################
#################################### Model ########################################
###################################################################################

from sklearn.ensemble import RandomForestClassifier

def load_player_model():

        ''' Load the compressed version of the Random Forest Model that will
        predict the player ID '''

        #with open('random_forest_classifier.pickle', 'rb') as f:
        #     player_model =  pickle.load(f)
        with open('model_compressed_1.pkl', 'rb') as f:
             player_model =  joblib.load(f)
        return player_model

#################################################################################

def get_features_vector(df_q, df_event):

    ''' Extract the aggregated feature vectors of the team (home or away). 

    This feature vectors basically consists in the aggregation for both team of all different 
    event occuring (related to one team at a time) and the pourcentage of positive outcome for 
    each event '''
    
    df_event['outcome'] = df_event['outcome'].astype(int)
    df = df_q.join(df_event.set_index('own_id')[['outcome', 'team_id','period_id']],on=['id_event'])
    
    #df['outcome'] = df['outcome'].astype(int)
    df_event['min'] = df_event['min'].astype(int)

    
    vec_list = []
    
    for team in range(2):
        
        #Choose the team (either 0 or 1)
        df_chunk = df[df['team_id'] == str(team)]
        #print(str(team))
        df_event_chunk =  df_event[df_event['team_id'] == str(team)]
        
        #Create empty vector
        vec_q = np.zeros(2*len(qualifier_ids)+4)
        vec_e = np.zeros(2*len(event_ids))

        # Get number of occurences
        dic_qualifier = dict(Counter(df_chunk.qualifier_id))
        
        # Get number of occurences
        dic_event = dict(Counter(df_event_chunk.type_id))
        
        # List of keys
        list_keys_q = list(dic_qualifier.keys())
        list_keys_event = list(dic_event.keys())
        #print('Qualifier')
        for feature in list_keys_q:
            
            # Get mapped index of the feature
            mapped = q_to_idx.get(feature,None)
            
            if mapped != None:
                
                #Get the list of outcome of the feature (% of successful event)
                l = list(df_chunk[df_chunk['qualifier_id']==feature].outcome)
                
                # If list isn't empty get the mean, and and to the vector : Nb_Occu, %success
                if l != []:
                    mean = np.round(np.mean(l),3)
                    vec_q[mapped*2+1] = mean
                vec_q[mapped*2] = dic_qualifier[feature]

            else : 
                pass
                #print('Q:',feature)
        
        l = list(df_chunk[df_chunk['qualifier_id']=='56'].value.values)  
        if len(l) == 0 :
            print('EMPTY LIST')
        else :
            c = Counter(l)
            vec_q[-1] = c.get('Right',0) / len(l)
            vec_q[-2] = c.get('Left',0)  / len(l)
            vec_q[-3] = c.get('Center',0)  / len(l)
            vec_q[-4] = c.get('Back',0)  / len(l)
        #56 : Back, Left, Center, Right

        # Same thing with the Event
        for feature in list_keys_event:
            mapped = event_to_idx.get(feature,None)
            
            if mapped != None:
                l = list(df_event_chunk[df_event_chunk['type_id']==feature].outcome)

                if l != []:
                    mean = np.round(np.mean(l),3)
                    vec_e[mapped*2+1] = mean
                vec_e[mapped*2] = dic_event[feature]
            else : 
                pass
                #print('Ev:',feature)

                
        # Add team (home/away), the average minute, and if it is second or first half
        vec = [int(team), int(df_event['min'].mean())/90, (df_event.period_id.unique()[0] == '2')]
        vec = np.concatenate([vec_q, vec_e, vec])
        vec.astype(float)
        vec_list.append(vec)
        
    return vec_list

#################################################################################

def get_features_vector_player(df_q, df_event, id_):

    ''' Extract the feature vectors of the player of interest. 

    This feature vectors basically consists in the aggregation of all different 
    event the player is involved in, and the pourcentage of positive outcome for 
    each event '''
    
    df_event['outcome'] = df_event['outcome'].astype(int)
    df = df_q.join(df_event.set_index('own_id')[['outcome', 'player_id','period_id']],on=['id_event'])
    
    #df['outcome'] = df['outcome'].astype(int)
    df_event['min'] = df_event['min'].astype(int)
    
    #Choose the player
    df_chunk = df[df['player_id'] == str(id_)]
    df_event_chunk =  df_event[df_event['player_id'] == str(id_)]
    

    #Create empty vector
    vec_q = np.zeros(2*len(qualifier_ids)+4)
    vec_e = np.zeros(2*len(event_ids))

    # Get number of occurences
    dic_qualifier = dict(Counter(df_chunk.qualifier_id))

    # Get number of occurences
    dic_event = dict(Counter(df_event_chunk.type_id))

    # List of keys
    list_keys_q = list(dic_qualifier.keys())
    list_keys_event = list(dic_event.keys())
    #print('Qualifier')
    for feature in list_keys_q:

        # Get mapped index of the feature
        mapped = q_to_idx.get(feature,None)

        if mapped != None:

            #Get the list of outcome of the feature (% of successful event)
            l = list(df_chunk[df_chunk['qualifier_id']==feature].outcome)

            # If list isn't empty get the mean, and and to the vector : Nb_Occu, %success
            if l != []:
                mean = np.round(np.mean(l),3)
                vec_q[mapped*2+1] = mean
            vec_q[mapped*2] = dic_qualifier[feature]

        else : 
            pass
            #print('Q:',feature)

    l = list(df_chunk[df_chunk['qualifier_id']=='56'].value.values)  
    if len(l) == 0 :
        print('EMPTY LIST PLAYER')
        #print(vec_q[-4:], df_event_chunk.shape )
    else :
        c = Counter(l)
        vec_q[-1] = c.get('Right',0) / len(l)
        vec_q[-2] = c.get('Left',0)  / len(l)
        vec_q[-3] = c.get('Center',0)  / len(l)
        vec_q[-4] = c.get('Back',0)  / len(l)
    #56 : Back, Left, Center, Right

    # Same thing with the Event
    for feature in list_keys_event:
        mapped = event_to_idx.get(feature,None)

        if mapped != None:
            l = list(df_event_chunk[df_event_chunk['type_id']==feature].outcome)

            if l != []:
                mean = np.round(np.mean(l),3)
                vec_e[mapped*2+1] = mean
            vec_e[mapped*2] = dic_event[feature]
        else : 
            pass
            #print('Ev:',feature)
    # Add team (home/away), the average minute, and if it is second or first half
    vec = [int(df_event_chunk.team_id.unique()[0]),int(df_event['min'].mean())/90, int(df_event.period_id.unique()[0] == '2')]
    vec = np.concatenate([vec_q, vec_e, vec])
    vec.astype(float)
        
    return vec


#################################################################################

def get_feature_vector(file):

    ''' Extract the final feature vector. Which is the concatenation of
    the team and player feature vectors '''
    
    game_df, event_df, q_df = parse_xml_file(file)

    event_df['min']= event_df['min'].astype(int)
    event_df['x']= event_df['x'].astype(float)
    event_df['y']= event_df['y'].astype(float)
    event_df['period_id']= event_df['period_id'].astype(int)
    
    team = event_df[event_df['player_id'] == '1']['team_id'].unique()
    #print(str(team[0]))
    vec_player = get_features_vector_player(q_df, event_df.loc[event_df.index[:-10],:],'1')
    vec = get_features_vector(q_df, event_df.loc[event_df.index[:-10],:])
    #print(vec[int(team[0])].shape, vec_player.shape)
    return np.concatenate([vec[int(team[0])],vec_player])

#################################################################################
