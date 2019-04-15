
''' main_psgx.py file that defines a unique function : Resultat().
It takes the path to an xml file and creat a .csv file with the 
four predicted values.'''

import numpy as np

###################################################################################
#################################### Packages #####################################
###################################################################################

from player_prediction import get_feature_vector, load_player_model#, player_names
player_model = load_player_model()

from event_prediction import build_model_team_prediction, get_array_file, load_model_x_y, get_array_file_position
team_model = build_model_team_prediction()
x_y_model = load_model_x_y()

from team_trick import filtered_indices_team, idx_to_player


###################################################################################

def Resultat(xml_file, output_file = 'ex.csv'):

    ''' From a XML file, it creates a CSV file with the 4 predicted values.
    Output_file : path/name.csv to be povided so as to create the csv file.'''

    ######## Player Prediction ########

    # Get feature vector for the player prediction
    vec_player = get_feature_vector(xml_file)
    filtered_indices = sorted(filtered_indices_team(xml_file))

    # Either we don't have any information on the team
    if len(filtered_indices) == 230 : 
    # Predict the id of the player
        res_player = player_model.predict(vec_player[np.newaxis,:])[0]

    # Or we can filter with player of the team
    else :
        res_player_list = player_model.predict_proba(vec_player[np.newaxis,:])[0]
        labels = list(player_model.classes_)
        filtered_labels = [labels.index(str(i)) for i in filtered_indices]
        id_filtered = np.argmax(res_player_list[filtered_labels])
        res_player = filtered_indices[id_filtered]

    # Find the related Player ID
    id_player = int(idx_to_player[int(res_player)])
    print(id_player)

    #print(idx_to_player[int(res_player)], player_names[idx_to_player[int(res_player)]])

    ######## Team Prediction ########

    vec_team, past_event_array = get_array_file(xml_file)
    next_team = int(float(team_model.predict([past_event_array[:,:,3],past_event_array[:,:,:3],vec_team])[0])>0.5)

    ######## XY Prediction ########
    vec_position = get_array_file_position(xml_file)
    y,x = x_y_model.predict(vec_position)[0]
    print(y,x)
    
    ######## Output ########

    resultat = [str(id_player), str(next_team), str(y),str(x)]
    resultat = str.join(', ',resultat)
    print(resultat)

    with open(output_file, 'w') as f:
        f.write(resultat)
        f.close()