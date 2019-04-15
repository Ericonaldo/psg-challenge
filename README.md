<center><img src="logo.png"  width="400"/></center>

# psg-challenge

Data Challenge X-PSG

Notebook : 
https://nbviewer.jupyter.org/github/dam-grassman/psg-challenge/blob/master/Step_by_step.ipynb

# Context

The goal of this challenge is to predict four different values out of an XML file that describes 15 min of a random game (from the second part of the season 2016/2017). All the teams and the players have been anonymised and can't be distinguised from one another except for one. 

- Predict the id of this unknonw player (out of all players that played at least 800 min during the first part of the season, ie around 230 players)
- Predict features of the next event : What team will it concern ? (Away or Home team) Where will be the ball during this event ? (Position X,Y)

To do so, we have at our disposition all the games that occur during the first part of the season. I've kept 4 differents matchdays (around 80 games) to test my prediction on unseen data.

At the end, i achieved around **21%** accuracy (id correct player predicted out of 230 possible choices) (CF Notebook)
