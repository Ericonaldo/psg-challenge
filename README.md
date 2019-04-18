<center><img src="images/logo.png"  width="400"/></center>

# psg-challenge

Data Challenge X-PSG

Notebook : 
https://nbviewer.jupyter.org/github/dam-grassman/psg-challenge/blob/master/Step_by_step.ipynb

# Context

The goal of this challenge is to predict four different values out of an XML file that describes 15 min of a random game (from the second part of the season 2016/2017). All the teams and the players have been anonymised and can't be distinguised from one another except for one. 

- Predict the id of this unknown particular player (out of all players that played at least 800 min during the first part of the season, ie around 230 players)
- Predict features of the next event : What team will it concern ? (Away or Home team) Where will take place this event ? (Position X,Y)

<center><img src="images/PSG_active_players.jpg"  width="400"/></center>

In the notebook (link above), i have tried to explain step by step my work in this challenge.

On ~20% of the database we had that I kept unseen, I get the following results (CF Notebook)

| Metrics     | Player    |Team  | X,Y     |
| --- | --- | --- | --- |
| Nb  Classes | 230       | 2    | linear  |
| Accuracy (%)|  21       |80    |         |
| MSE\MAE     |           |      |  536\17 |

## Configuration 

- keras=2.2.4
- tensorflow-gpu=1.12.0
- numpy=1.15.4
- pandas=0.23.4
- tqdm=4.31.1
- joblib=0.13.2 
- scikit-learn=0.20.2
- lxml

## Files 

- **pickle** : bunch of dictionnary saved as pickle files (time_per_player.pickle, team_squad_active.pickle, team_squad.pickle, qualifier_ids.pickle, event_ids.pickle).

- **h5** : weights of neural networks model to predict the team of the next event (team_prediction.h5) as well as the position of the next event (x_y.h5).

- **model_compressed_1.joblib**: compressed weights and object structure of the Random Forest Classifier that we use to predict the unknown player.

## Scripts   

- **main_psgx.py** : a unique function Resultat() that takes in input the path to a xml file and return 4 values : a player id, the next team (0 or 1) and the position y and x.

- **install_psgx.py** : installations files with os.system instructions to install all packages.

- **event_prediction.py** :  all functions related to the prediction of the next event (team and position).

- **player_prediction.py** :  all functions related to the prediction of the unknown player.

- **team_trick.py** : short script that deals with the team trick I'm referring to in the notebook.

- **load_data.py** : functions related to the xml parser to transform xml files into dataframes.
