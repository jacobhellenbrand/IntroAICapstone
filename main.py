import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import powerlaw
import numpy as np
from mpmath import degree

df = pd.read_csv('player_play.csv')
def build_network(df, conditions, event_weights):
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
                            edge_weight = event_weights.get((player1_var, player2_var), 0)

                            if graph.has_edge(player1, player2):
                                graph[player1][player2]['weight'] += edge_weight
                            else:
                                graph.add_edge(player1, player2, weight=edge_weight)
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

# Dictionary of event pairs
event_pairs = {
    ('hadDropback', 'hadPassReception'): 3,
    ('hadDropback', 'wasTargettedReceiver'): 1,
    ('hadPassReception', 'soloTackle'): 2,
    ('hadPassReception', 'tackleAssist'): 1.5,
    ('hadRushAttempt', 'soloTackle'): 3,
    ('hadRushAttempt', 'tackleAssist'): 1.5,
    ('hadDropback', 'hadInterception'): 3,
    ('fumbleLost', 'fumbleRecoveries'): 2,
    ('fumbleLost', 'forcedFumbleAsDefense'): 6,
    ('hadDropback', 'causedPressure'): 3,
    ('pressureAllowedAsBlocker', 'causedPressure'): 2,
    ('hadDropback', 'passDefensed'): 3,
    ('wasTargettedReceiver', 'passDefensed'): 3,
    ('hadDropback', 'quarterbackHit'): 4,
    ('hadDropback', 'sackYardsAsDefense'): 4,
    ('tackleAssist', 'tackleAssist'): 2.5,
    ('forcedFumbleAsDefense', 'fumbleRecoveries'): 7,
    ('forcedFumbleAsDefense', 'hadPassReception'): 6,
    ('forcedFumbleAsDefense', 'hadRushAttempt'): 6,
    ('hadDropback', 'tackleAssist'): 2,
    ('hadDropback', 'soloTackle'): 3,
}

# Dictionary of event values
event_values = {
    'Rush Attempt': 73+7,
    'hadDropback': 184,
    'wasTargettedReceiver': 37+10,
    'hadPassReception': 27+8,
    'soloTackle': 23,
    'tackleAssist': 14,
    'hadInterception': 2 + 18.4 + 7.3 + 2.7,
    'fumbleLost': 2+ 18.4 + 7.3 + 2.7,
    'fumbleRecoveries': 1+ 18.4 + 7.3 + 2.7,
    'forcedFumbleAsDefense': 1 + 18.4 + 7.3 + 2.7,
    'causedPressure': 10 + 18.4 + 7.3 + 2.7,
    'pressureAllowedAsBlocker': 16,
    'passDefensed': 3 + 18.4 + 7.3 + 2.7,
    'quarterbackHit': 4 + 18.4 + 7.3 + 2.7,
    'sackYardsAsDefense': 18 + 18.4 + 7.3 + 2.7,
}

# Calculate new values for each pair
updated_event_pairs = {}
for (event1, event2), _ in event_pairs.items():
    value1 = event_values.get(event1, 0)  # Get the value for the first event
    value2 = event_values.get(event2, 0)  # Get the value for the second event
    if value1 + value2 != 0:
        new_value = 10 / (value1 + value2)  # Calculate the new value
    else:
        new_value = 0  # Avoid division by zero
    updated_event_pairs[(event1, event2)] = new_value

# Print updated dictionary
for key, value in updated_event_pairs.items():
    print(f"{key}: {value:.3f}")



conditions = [
    ('hadDropback', 'hadPassReception'),
    ('hadDropback', 'wasTargettedReceiver'),
    ('hadPassReception', 'soloTackle'),
    ('hadPassReception', 'tackleAssist'),
    ('hadRushAttempt', 'soloTackle'),
    ('hadRushAttempt', 'tackleAssist'),
    ('hadDropback', 'hadInterception'),
    ('fumbleLost', 'fumbleRecoveries'),
    ('fumbleLost', 'forcedFumbleAsDefense'), #maybe add forced fumble with rush attempt/pass recep
    ('forcedFumbleAsDefense', 'hadPassReception'), #new
    ('forcedFumbleAsDefense', 'hadRushAttempt'), #new
    ('hadDropback', 'causedPressure'),
    ('pressureAllowedAsBlocker', 'causedPressure'),
    ('hadDropback', 'passDefensed'),
    ('wasTargettedReceiver', 'passDefensed'),
    ('hadDropback', 'quarterbackHit'),
    ('hadDropback', 'sackYardsAsDefense'),
    ('hadDropback', 'tackleAssist'), #new
    ('hadDropback', 'soloTackle'), #new
    ('tackleAssist', 'tackleAssist'),
    ('forcedFumbleAsDefense', 'fumbleRecoveries')
]

inverted_conditions = [(y, x) for x, y in conditions]
all_conditions = conditions + inverted_conditions

symmetric_event_weights = updated_event_pairs.copy()
for (a, b), weight in updated_event_pairs.items():
    symmetric_event_weights[(b, a)] = weight

player_network = build_network(df, all_conditions, symmetric_event_weights)


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
    """
                                                #weight='weight'
    weighted_degrees = [degree for _, degree in graph.degree(weight='weight')]
    return weighted_degrees


def visualize_degree_distribution(degrees, name):
    """
    Visualize the degree distribution on linear and log-log scales.
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
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.title(f'Log-Log Degree Distribution of {name}')
    plt.show()

def visualize_cumulative_degree_distribution(degrees, name):
    """
    Visualize the cumulative degree distribution on linear and log-log scales.
    """
    unique_degrees, counts = np.unique(degrees, return_counts=True)

    cumulative_counts = np.cumsum(counts[::-1])[::-1]
    cumulative_distribution = cumulative_counts / sum(counts)

    # Linear scale
    plt.figure(figsize=(8, 6))
    plt.plot(unique_degrees, cumulative_distribution, 'o')
    plt.xlabel("Degree")
    plt.ylabel("Cumulative Fraction of Nodes")
    plt.title(f'Cumulative Degree Distribution of {name}')
    plt.show()

    # Log-log scale
    plt.figure(figsize=(8, 6))
    plt.loglog(unique_degrees, cumulative_distribution, 'o')
    plt.xlabel("Degree")
    plt.ylabel("Cumulative Fraction of Nodes")
    plt.title(f'Log-Log Cumulative Degree Distribution of {name}')
    plt.show()


def fit_power_law(degrees, name):
    """
    Fit a power law to the degree distribution and print alpha and xmin values.
    """
    fit = powerlaw.Fit(degrees, discrete=True)
    print(f"The alpha value for {name}: {fit.alpha}")
    print(f"The xmin value for {name}: {fit.xmin}")

    plt.figure(figsize=(8, 6))
    fit.plot_pdf(label='Empirical Data', color='blue')
    fit.power_law.plot_pdf(label='Power Law Fit', color='orange')
    plt.xlabel("Degree")
    plt.ylabel("PDF")
    plt.legend()
    plt.title(f'Power Law Fit for {name}')
    plt.show()

degrees = get_degree_distribution(player_network)
visualize_degree_distribution(degrees, "Player Network")
visualize_cumulative_degree_distribution(degrees, "Player Network")
fit_power_law(degrees, "Player Network")