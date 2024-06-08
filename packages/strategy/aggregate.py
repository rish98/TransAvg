# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Aggregation functions for strategy implementations."""


from functools import reduce
from typing import List, Tuple


from flwr.common import NDArray, NDArrays
from sklearn.decomposition import PCA
from flwr.server.strategy import trans_autoencoder
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

def assign_client_scores_exp_decay(mse_losses, max_loss,min_loss,drop_rate):
    """
    Assign client scores based on an exponential decay function of MSE losses.
    Scores are 1.0 for MSE losses less than 15.
    Scores decrease at a decreasing rate for MSE losses between 15 and 100.
    Scores are 0 for MSE losses greater than 100.
    """
    client_scores = []
    for loss in mse_losses:
        if loss < min_loss:
            score = 1.0
        elif min_loss <= loss <= max_loss:
            score = np.exp(-drop_rate * (loss - min_loss) ** 2 / (max_loss - min_loss) ** 2)
        else:
            score = 0.0
        client_scores.append(score)
    return client_scores


def trans_aggregate(results: List[Tuple[NDArrays, int]],last_layers:List[NDArrays],clients_in_round:List[int],curr_round ) -> NDArrays:
    """Compute weighted average."""

    print("in",len(last_layers),len(last_layers[0]))

    test= np.asarray(last_layers)
    print(test.shape)
    pca = PCA(n_components=9)

    pca_res_np=pca.fit_transform(test)
    # explained_var_ratio = pca.explained_variance_ratio_
    # total_var_ratio = np.sum(explained_var_ratio)
    # print("Explained variance ratio of each component:", explained_var_ratio)
    # print("Total explained variance:", total_var_ratio)

    pca_res_torch=torch.from_numpy(pca_res_np)
    # print(pca_res_torch.shape)

    # Calculate the MSE loss
    # mse_loss = nn.MSELoss()(reconstructed_weights, pca_res_np)
    # print(mse_loss)
    # print(mse_loss[0])

    model_path = "./trans_model.pth"
    
    if os.path.exists(model_path):
        model = torch.load(model_path)
        # print("Loaded model from:", model_path)
    else:
        # Create a new model if the file doesn't exist
        model = trans_autoencoder.Autoencoder(input_dim=pca_res_np.shape[1], hidden_dim=64, num_layers=2, nhead=3, dim_feedforward=256, dropout=0.1)


    reconstructed_weights = model(pca_res_np)
    trans_autoencoder.train_autoencoder(model, pca_res_np, 100, 5, 0.0001)
    torch.save(model, model_path)


    round_mseloss=[nn.MSELoss()(reconstructed_weights[i], pca_res_torch[i]).item() for i in range(pca_res_torch.shape[0])]
    client_mseloss=[(a, f'{b:.4f}') for a, b in zip(clients_in_round, round_mseloss)]
    print(client_mseloss)
    client_scores = assign_client_scores_exp_decay(round_mseloss, max_loss=(max(round_mseloss)),min_loss=(min(round_mseloss)), drop_rate=1)
    client_scores_f=[(a, f'{b:.4f}') for a, b in zip(clients_in_round, client_scores)]
    
    total_score = sum(client_scores)
    num_examples_total = sum([num_examples for _, num_examples in results])

    # client_scores_prev = load_data('./client_scores.txt')
    # client_scores_dict={}
    # for client, score in zip(clients_in_round, client_scores):
    #     client_scores_dict[client] = score
    # print("prev",client_scores_prev,"cirr",client_scores_dict)
    # client_scores_dict.update(client_scores_prev)
    # store_data(client_scores_dict, './client_scores.txt', curr_round)
    client_scores_f = sorted(client_scores_f, key=lambda x: x[0])
    print(client_scores_f)

    # if curr_round in [10,20,30,40,50,60,70,80,90,100]:
    #     with open('./client_scores.txt', "a") as f:
    #         f.write(f"Round {curr_round}:   {client_scores_f}\n")

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]

    weighted_weights = [
        [layer * score for layer in weights] for weights, score in zip([w for w, _ in results],client_scores)
    ]
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / total_score
        for layer_updates in zip(*weighted_weights)
    ]


    return weights_prime

def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])
    print()
    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def aggregate_median(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute median."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute median weight of each layer
    median_w: NDArrays = [
        np.median(np.asarray(layer), axis=0) for layer in zip(*weights)  # type: ignore
    ]
    return median_w


def aggregate_krum(
    results: List[Tuple[NDArrays, int]], num_malicious: int, to_keep: int
) -> NDArrays:
    """Choose one parameter vector according to the Krum fucntion.

    If to_keep is not None, then MultiKrum is applied.
    """
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute distances between vectors
    distance_matrix = _compute_distances(weights)

    # For each client, take the n-f-2 closest parameters vectors
    num_closest = max(1, len(weights) - num_malicious - 2)
    closest_indices = []
    for i, _ in enumerate(distance_matrix):
        closest_indices.append(
            np.argsort(distance_matrix[i])[1 : num_closest + 1].tolist()  # noqa: E203
        )

    # Compute the score for each client, that is the sum of the distances
    # of the n-f-2 closest parameters vectors
    scores = [
        np.sum(distance_matrix[i, closest_indices[i]])
        for i in range(len(distance_matrix))
    ]

    if to_keep > 0:
        # Choose to_keep clients and return their average (MultiKrum)
        best_indices = np.argsort(scores)[::-1][len(scores) - to_keep :]  # noqa: E203
        best_results = [results[i] for i in best_indices]
        return aggregate(best_results)

    # Return the index of the client which minimizes the score (Krum)
    return weights[np.argmin(scores)]


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def aggregate_qffl(
    parameters: NDArrays, deltas: List[NDArrays], hs_fll: List[NDArrays]
) -> NDArrays:
    """Compute weighted average based on  Q-FFL paper."""
    demominator = np.sum(np.asarray(hs_fll))
    scaled_deltas = []
    for client_delta in deltas:
        scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])
    updates = []
    for i in range(len(deltas[0])):
        tmp = scaled_deltas[0][i]
        for j in range(1, len(deltas)):
            tmp += scaled_deltas[j][i]
        updates.append(tmp)
    new_parameters = [(u - v) * 1.0 for u, v in zip(parameters, updates)]
    return new_parameters


def _compute_distances(weights: List[NDArrays]) -> NDArray:
    """Compute distances between vectors.

    Input: weights - list of weights vectors
    Output: distances - matrix distance_matrix of squared distances between the vectors
    """
    flat_w = np.array(
        [np.concatenate(p, axis=None).ravel() for p in weights]  # type: ignore
    )
    distance_matrix = np.zeros((len(weights), len(weights)))
    for i, _ in enumerate(flat_w):
        for j, _ in enumerate(flat_w):
            delta = flat_w[i] - flat_w[j]
            norm = np.linalg.norm(delta)  # type: ignore
            distance_matrix[i, j] = norm**2
    return distance_matrix
