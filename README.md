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
