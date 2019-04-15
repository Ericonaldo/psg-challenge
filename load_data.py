''' Short script that contains the data loader function '''


###################################################################################
#################################### Packages #####################################
###################################################################################

from lxml import etree
from lxml import objectify

import pandas as pd
import numpy as np
import os

event_df_columns = ['id_game','own_id','id', 'event_id', 'type_id', 'period_id', 'min',  'sec', 'player_id', 'team_id',
                    'outcome', 'x', 'y', 'timestamp', 'last_modified', 'version']

game_df_columns = ['id','away_score','away_team_id','away_team_name','competition_id','competition_name','game_date',
                   'home_score', 'home_team_id', 'home_team_name', 'matchday', 'period_1_start', 'period_2_start', 
                   'season_id', 'season_name']

q_df_columns = ['id_event','id','qualifier_id','value']

###################################################################################
############################## Data Loader ########################################
###################################################################################

def parse_xml_file(file):
    
    event_dictionnary = []
    q_dictionnary = []
    game_dictionnary = dict()
    tree = etree.parse(file)
    assert len(tree.xpath("/Games/Game")) == 1
    game = tree.xpath("/Games/Game")[0]
    game_id = game.attrib['id']

    game_dictionnary[game_id] = [game.attrib.get(key,'') for key in game_df_columns]
    
    for id_event,event in enumerate(tree.xpath("/Games/Game/Event")):
        try : 
            event_list = [game_id, id_event]
            for feature in event_df_columns[2:]: 
                event_list.append(event.attrib.get(feature,''))
            event_dictionnary.append(event_list)
            #event_df = event_df.append(pd.Series(event_list,   index = event_df_columns),  ignore_index = True)
        except : 
            print(event, event.keys())
        #print(id_event)
        event_id = event_list[1]
        for i in range(len(event.getchildren())):
            q = event.getchildren()[i].attrib
            q_dictionnary.append([id_event]+[q.get(key,'') for key in q_df_columns[1:]])
    #print(event_dictionnary[-10:]) 
    event_df = pd.DataFrame(event_dictionnary,columns=event_df_columns)
    game_df = pd.DataFrame.from_dict(game_dictionnary, orient='index', columns = game_df_columns)
    
    q_df = pd.DataFrame(q_dictionnary,columns=q_df_columns)

    return game_df, event_df, q_df