import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
#
# df = pd.read_csv('player_play.csv')
# def build_network(df, game_id, conditions):
#     game_df = df[df['gameId'] == game_id]
#     graph = nx.Graph()
#
#     for play_id, play_data in game_df.groupby('playId'):
#         player_data = {condition: None for condition in conditions}
#
#         for _, row in play_data.iterrows():
#             for player1_var, player2_var in conditions:
#                 if player_data[(player1_var, player2_var)] is None:
#                     if is_condition_met(row, player1_var):
#                         player_data[(player1_var, player2_var)] = row['nflId']
#
#                 if player_data[(player1_var, player2_var)] is not None:
#                     if is_condition_met(row, player2_var):
#                         player2 = row['nflId']
#                         player1 = player_data[(player1_var, player2_var)]
#                         if player1 != player2:
#                             graph.add_edge(player1, player2)
#     return graph
#
#
# def is_condition_met(row, variable):
#     """
#     Check if the given condition (column) is met. This handles both booleans and numeric conditions.
#     """
#     value = row.get(variable)
#
#     print(f"Checking condition for {variable}: {value}")
#     if isinstance(value, bool):
#         return value is True
#     elif isinstance(value, (int, float)):
#         return pd.notna(value) and value >= 1
#     return False
#
# conditions = [
#     ('hadDropback', 'hadPassReception'),
#     ('hadDropback', 'wasTargettedReceiver'),
#     ('hadPassReception', 'soloTackle'),
#     ('hadPassReception', 'tackleAssist'),
#     ('hadRushAttempt', 'soloTackle'),
#     ('hadRushAttempt', 'tackleAssist'),
#     ('hadDropback', 'hadInterception'),
#     ('fumbleLost', 'fumbleRecoveries'),
#     ('fumbleLost', 'forcedFumbleAsDefense'),
#     ('hadDropback', 'causedPressure'),
#     ('pressureAllowedAsBlocker', 'causedPressure'),
#     ('hadDropback', 'passDefensed'),
#     ('wasTargettedReceiver', 'passDefensed'),
#     ('hadDropback', 'quarterbackHit'),
#     ('hadDropback', 'sackYardsAsDefense'),
#     ('tackleAssist', 'tackleAssist'),
#     ('forcedFumbleAsDefense', 'fumbleRecoveries'),
#     # ('causedPressure', 'hadInterception'),
#     # ('hadRushAttempt', 'blockedPlayerNFLId1'),
#     # ('hadRushAttempt', 'blockedPlayerNFLId2'),
#     # ('hadRushAttempt', 'blockedPlayerNFLId3'),
# ]
#
# game_id = 2022090800
# player_network = build_network(df, game_id, conditions)
#
# print(f"Player Interaction Network: {len(player_network.nodes)} nodes, {len(player_network.edges)} edges")
# nx.draw(player_network, with_labels=True, node_size=500, node_color="orange", font_size=8)
# plt.title("Player Interaction Network")
# plt.show()
#
# player_df = pd.read_csv('players.csv')
#
# for _, player_row in player_df.iterrows():
#     nfl_id = player_row['nflId']
#     if nfl_id in player_network.nodes:
#         # Add player attributes (e.g., name, position) to the node
#         player_network.nodes[nfl_id]['name'] = player_row['displayName']
#         player_network.nodes[nfl_id]['position'] = player_row['position']
#
# downloads_path = os.path.expanduser('~/Desktop/Math-479')
# node_data = [(nfl_id, data['name'], data['position']) for nfl_id, data in player_network.nodes(data=True)]
# node_data_df = pd.DataFrame(node_data, columns=['ID', 'Label', 'Position'])
# node_data_df.to_csv(os.path.join(downloads_path, 'player_network_nodes.csv'), index=False)
# edges_data = [(player1, player2) for player1, player2 in player_network.edges()]
# # Create a DataFrame for edges with 'Source' and 'Target' columns for Gephi
# edges_data_df = pd.DataFrame(edges_data, columns=['Source', 'Target'])
# # Save the edge data in Gephi-compatible CSV format
# edges_data_df.to_csv(os.path.join(downloads_path, 'player_network_edges.csv'), index=False)
# # Confirm that the data was saved
# print("Player network nodes and edges have been saved to 'player_network_nodes.csv' and 'player_network_edges.csv'.")



df = pd.read_csv('player_play.csv')

def build_network(df, game_id, conditions):
    game_df = df[df['gameId'] == game_id]
    graph = nx.Graph()

    # Dictionary to store edge weights
    edge_weights = {}

    for play_id, play_data in game_df.groupby('playId'):
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
                        if player1 != player2:
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

game_id = 2022090800
player_network = build_network(df, game_id, conditions)

print(f"Player Interaction Network: {len(player_network.nodes)} nodes, {len(player_network.edges)} edges")
nx.draw(player_network, with_labels=True, node_size=500, node_color="orange", font_size=8)
plt.title("Player Interaction Network")
plt.show()

player_df = pd.read_csv('players.csv')

for _, player_row in player_df.iterrows():
    nfl_id = player_row['nflId']
    if nfl_id in player_network.nodes:
        # Add player attributes (e.g., name, position) to the node
        player_network.nodes[nfl_id]['name'] = player_row['displayName']
        player_network.nodes[nfl_id]['position'] = player_row['position']

downloads_path = os.path.expanduser('~/Desktop/Math-479')

# Prepare node data for CSV export
node_data = [(nfl_id, data['name'], data['position']) for nfl_id, data in player_network.nodes(data=True)]
node_data_df = pd.DataFrame(node_data, columns=['ID', 'Label', 'Position'])
node_data_df.to_csv(os.path.join(downloads_path, 'player_network_nodes.csv'), index=False)

# Prepare edges data with 'Source', 'Target' and 'Weight' for Gephi export
edges_data = [(player1, player2, data['weight']) for player1, player2, data in player_network.edges(data=True)]
edges_data_df = pd.DataFrame(edges_data, columns=['Source', 'Target', 'Weight'])

# Save the edge data in Gephi-compatible CSV format with weights
edges_data_df.to_csv(os.path.join(downloads_path, 'player_network_edges.csv'), index=False)

# Confirm that the data was saved
print("Player network nodes and edges have been saved to 'player_network_nodes.csv' and 'player_network_edges.csv'.")