''' Short Script where we take avantage of the availibility of the team squads.

Indeed, some player_id might be available during substitutions for instance, and one 
might recover the team playing. This will allow us to filter our predictions by looking 
only at the player within the predicted team. '''


#################################################################################
################################ Functions ######################################
#################################################################################

import pickle
from load_data import parse_xml_file

# with open('player_names.pickle', 'rb') as handle:
#     player_names = pickle.load(handle)

with open('time_per_player.pkl', 'rb') as handle:
    dictionnary_time_per_player = pickle.load(handle)

active_player_800 = [player_id for player_id, value in dictionnary_time_per_player.items() if value >= 780]
#print('{} active players that played more than 800 minutes'.format(len(active_player_800)))

player_to_idx = {}
for i,j in enumerate(active_player_800):
    player_to_idx[str(j)] = i

idx_to_player = {}
for i,j in enumerate(active_player_800):
    idx_to_player[i] = str(j)

# load team squad
with open('team_squad.pkl', 'rb') as handle:
    dic_team =  pickle.load(handle)
    
# load active player of each team
with open('team_squad_active.pkl', 'rb') as handle:
    dic_team_active =  pickle.load(handle)
    
list_team = sorted(dic_team.keys())

team_idx = {}
for i,j in enumerate(list_team):
    team_idx[j] = i 

# with open('team_names.pickle', 'rb') as handle:
#     team_name = pickle.load(handle)
    

def get_team(set_player):

    ''' Get the best match between the available ids and the
    team squad '''

    best_match, inter = '', 0
    for team in dic_team.keys():
        inter_list = dic_team[team].intersection(set_player)
        if len(inter_list) > inter :
            inter = len(inter_list)
            best_match = team
    return best_match, inter


def get_team_id(xml_file):

    ''' Get the id of teams playing '''
    
    game_df, event_df, q_df = parse_xml_file(xml_file)
    
    q_id_with_players = [7, 30 , 53, 194, 281]
    teams= {}

    for i in q_id_with_players:
        df_chunk =  q_df[q_df['qualifier_id']==str(i)] 
        for row in df_chunk.iterrows():
            id_event = row[1]['id_event']
            value = row[1]['value'].split(', ')
            #print(id_event,value)
            id_team = event_df_test[event_df_test['own_id'] ==  id_event].team_id.values[0]
            teams[id_team] = teams.get(id_team,[]) + value
    
    team_dic = {}
    for t in range(2) : 
        teams[t] = set(teams[str(t)])
        team, val = get_team(teams[t])
        if val > 1 :
            team_dic[str(t)] = team
        #print(team_name[team])
    return team_dic

def player_team_id(xml_file):

    ''' Get the id of the player of interest '''

    team_playing = get_team_id(test_file)
    
    game_df, event_df, q_df = parse_xml_file(xml_file)
    team_id = event_df[event_df['player_id'] == '1'].team_id.unique()[0]
    return team_playing.get(team_id, 'None')

#################################################################################

def filtered_indices_team(xml_file): 


    ''' Put all three previous functions all together. From a xml file, 
    one get, if available, the list of indices to filter on when predicted the
    player identity by restrecting on the active player of a specific team '''

    game_df, event_df, q_df = parse_xml_file(xml_file)
    
    q_id_with_players = [7, 30 , 53, 194, 281]
    teams= {}

    for i in q_id_with_players:
        df_chunk =  q_df[q_df['qualifier_id']==str(i)] 
        for row in df_chunk.iterrows():
            id_event = row[1]['id_event']
            value = row[1]['value'].split(', ')
            #print(id_event,value)
            id_team = event_df[event_df['own_id'] ==  id_event].team_id.values[0]
            teams[id_team] = teams.get(id_team,[]) + value
    
    team_dic = {}
    for t in range(2) : 
        teams[t] = set(teams.get(str(t),[]))
        team, val = get_team(teams[t])
        if val > 1 :
            team_dic[str(t)] = team

    game_df, event_df, q_df = parse_xml_file(xml_file)
    team_id = event_df[event_df['player_id'] == '1'].team_id.unique()[0]
    player_team = team_dic.get(team_id, 'None')

    if player_team == 'None':
        return [i for i in range(230)]

    else : 
        idx_to_filter = [player_to_idx[i] for i in dic_team_active[player_team]]
        return idx_to_filter

