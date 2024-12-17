# NFL In-Game Player Interaction Network Analysis

### By: Jacob Hellenbrand

## The Code

* This Python code takes a look at evaluating and creating a weighted undirected network
between NFL players interacting with the emphasis of interaction being around the ball. 


## The Data

* The data comes from the Big Data Bowl hosted by the NFL [here](https://www.kaggle.com/competitions/nfl-big-data-bowl-2024). 
The main files used from it are in the repository and are player_plays, and players.
As well the plays csv is listed for possible future work. 


## Supporting Files

* In the repository there were some supporting files that helped with the analysis of this network. 
The first of them being an RStudio file that helped to analyze important nodes in the network. If you download
the file and have it in the same folder as the csv's the code should be able to run non problematically. 
  * The python code will need to be ran before using the RStudio file, as it creates 2 necessary csv's
    * Edit line 176 to your devices correct download path
  * Additionally, you will need to download the playerNetwork3.csv to have the code run, see below. 
  * There is also code in the RStudio file that aids in adding useful information such as grouped positions, offense/defense
  terms, and adds the players teams to their row. 

* Also there is a gephi file that should be used when doing network analysis. This has the primary ranking of the nodes
and it is where a file named playerNetwork3.csv was created from. 