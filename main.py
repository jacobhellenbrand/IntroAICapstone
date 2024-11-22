import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import powerlaw
import numpy as np

df = pd.read_csv('player_play.csv')

def build_network(df, conditions):
    graph = nx.Graph()

    for (game_id, play_id), play_data in df.groupby(['gameId', 'playId']):
        player_data = {condition: None for condition in conditions}

        for _, row in play_data.iterrows():
            for player1_var, player2_var in conditions:
                if player_data[(player1_var, player2_var)] is None:
                    if is_condition_met(row, player1_var):
                        player_data[(player1_var, player2_var)] = row['nflId']

                if player_data[(player1_var, player2_var)] is not None:
                    if is_condition_met(row, player2_var):
                        player2 = row['nflId']
                        player1 = player_data[(player1_var, player2_var)]

                        if player1 is not None and player2 is not None and player1 != player2:
                            # Increment edge weight for each occurrence of the link
                            if graph.has_edge(player1, player2):
                                graph[player1][player2]['weight'] += 1
                            else:
                                graph.add_edge(player1, player2, weight=1)
    return graph


def is_condition_met(row, variable):
    """
    Check if the given condition (column) is met. This handles both booleans and numeric conditions.
    """
    value = row.get(variable)

    print(f"Checking condition for {variable}: {value}")
    if isinstance(value, bool):
        return value is True
    elif isinstance(value, (int, float)):
        return pd.notna(value) and value >= 1
    return False

conditions = [
    ('hadDropback', 'hadPassReception'),
    ('hadDropback', 'wasTargettedReceiver'),
    ('hadPassReception', 'soloTackle'),
    ('hadPassReception', 'tackleAssist'),
    ('hadRushAttempt', 'soloTackle'),
    ('hadRushAttempt', 'tackleAssist'),
    ('hadDropback', 'hadInterception'),
    ('fumbleLost', 'fumbleRecoveries'),
    ('fumbleLost', 'forcedFumbleAsDefense'),
    ('hadDropback', 'causedPressure'),
    ('pressureAllowedAsBlocker', 'causedPressure'),
    ('hadDropback', 'passDefensed'),
    ('wasTargettedReceiver', 'passDefensed'),
    ('hadDropback', 'quarterbackHit'),
    ('hadDropback', 'sackYardsAsDefense'),
    ('tackleAssist', 'tackleAssist'),
    ('forcedFumbleAsDefense', 'fumbleRecoveries'),
]

inverted_conditions = [(y, x) for x, y in conditions]
all_conditions = conditions + inverted_conditions

game_id = 2022090800
player_network = build_network(df, all_conditions)

print(f"Player Interaction Network: {len(player_network.nodes)} nodes, {len(player_network.edges)} edges")
nx.draw(player_network, with_labels=True, node_size=500, node_color="orange", font_size=8)
plt.title("Player Interaction Network")
plt.show()

player_df = pd.read_csv('players.csv')

for _, player_row in player_df.iterrows():
    nfl_id = player_row['nflId']
    if nfl_id in player_network.nodes:
        # Add player name/position to the node
        player_network.nodes[nfl_id]['name'] = player_row['displayName']
        player_network.nodes[nfl_id]['position'] = player_row['position']

downloads_path = os.path.expanduser('~/Desktop/Math-479')
node_data = [(nfl_id, data['name'], data['position']) for nfl_id, data in player_network.nodes(data=True)]
node_data_df = pd.DataFrame(node_data, columns=['ID', 'Label', 'Position'])
node_data_df.to_csv(os.path.join(downloads_path, 'player_network_nodes.csv'), index=False)
edges_data = [(player1, player2, data['weight']) for player1, player2, data in player_network.edges(data=True)]
edges_data_df = pd.DataFrame(edges_data, columns=['Source', 'Target', 'Weight'])
edges_data_df.to_csv(os.path.join(downloads_path, 'player_network_edges.csv'), index=False)

print("Player network nodes and edges have been saved to 'player_network_nodes.csv' and 'player_network_edges.csv'.")


def get_degree_distribution(graph):
    """
    Extracts the degree distribution from a NetworkX graph.

    param:
        graph: A NetworkX graph object
    return:
        A list of degrees for each node
    """
    degrees = [degree for _, degree in graph.degree()]
    return degrees

def visualize_degree_distribution(degrees, name):
    """
    Visualize the degree distribution on linear and log-log scales.

    param:
        degrees: List of node degrees
        name: Title for the plots
    """
    # Unique degree values and their counts
    unique_degrees = np.unique(degrees)
    counts = [degrees.count(d) for d in unique_degrees]

    # Linear scale
    plt.figure(figsize=(8, 6))
    plt.plot(unique_degrees, counts, 'o')
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.title(f'Degree Distribution of {name}')
    plt.show()

    # Log-log scale
    plt.figure(figsize=(8, 6))
    plt.loglog(unique_degrees, counts, 'o')
    plt.xlabel("Degree (log scale)")
    plt.ylabel("Number of Nodes (log scale)")
    plt.title(f'Log-Log Degree Distribution of {name}')
    plt.show()


def fit_power_law(degrees, name):
    """
    Fit a power law to the degree distribution and print alpha and xmin values.

    param:
        degrees: List of node degrees
        name: Title for the dataset
    """
    fit = powerlaw.Fit(degrees, discrete=True)
    print(f"The alpha value for {name}: {fit.alpha}")
    print(f"The xmin value for {name}: {fit.xmin}")

    # Plot the power-law fit
    plt.figure(figsize=(8, 6))
    fit.plot_pdf(label='Empirical Data', color='blue')
    fit.power_law.plot_pdf(label='Power Law Fit', color='orange')
    plt.xlabel("Degree")
    plt.ylabel("PDF")
    plt.legend()
    plt.title(f'Power Law Fit for {name}')
    plt.show()

degrees = get_degree_distribution(player_network)

# Visualize degree distribution
visualize_degree_distribution(degrees, "Player Network")

# Fit and analyze power law
fit_power_law(degrees, "Player Network")