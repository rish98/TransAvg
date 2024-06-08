import hydra 
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset,partition_data_dirichlet
from client import generate_client_fn
import flwr as fl
from flwr.server.strategy.krum import Krum
from flwr.server.strategy.trans_avg import TransAvg
# from flwr.server.strategy.fedtrimmedavg import FedTrimmedAvg
from server import get_on_fit_config, get_evaluate_fn
import torch 
torch.cuda.empty_cache()


import warnings
warnings.filterwarnings("ignore")


# x round 100, acc y plot 3 attacks 
# same attack with full vs bin
# contin learnig 
# weight filtering 


# differentiate 
# loss function - after recons 
# decoder - binary, sigmoid 
# contlearn label new, retraining wo forgeting 


from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

import pickle
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

# from flwr.server.strategy.trans_avg import TransAvg


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
        
    #for iid
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

    #for noniid
    # trainloaders, validationloaders, testloader = partition_data_dirichlet(
    #     cfg.num_clients,0.5, cfg.batch_size
    # )
    
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    # choose from fl.server.strategy.FedAvg,fl.server.strategy.FedMedian,Krum, TransAvg
    strategy = TransAvg(
        fraction_fit=0.0001,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
        fraction_evaluate=0.0001,  # similar to fraction_fit, we don't need to use this argument.
        min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        min_available_clients=cfg.num_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
        
    )  # a function to run on the server side to evaluate the global model.

    
    # With the dataset partitioned, the client function and the strategy ready, we can now launch the simulation!
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        strategy=strategy,  # (optional) controls the degree of parallelism of your simulation.
        # Lower resources per client allow for more clients to run concurrently
        # (but need to be set taking into account the compute/memory footprint of your run)
        # `num_cpus` is an absolute number (integer) indicating the number of threads a client should be allocated
        # `num_gpus` is a ratio indicating the portion of gpu memory that a client needs.
        client_resources={"num_cpus":2,"num_gpus":0.2}
    )
    # save_path = HydraConfig.get().runtime.output_dir
    # results_path = Path(save_path) / "results.pkl"

    # # add the history returned by the strategy into a standard Python dictionary
    # # you can add more content if you wish (note that in the directory created by
    # # Hydra, you'll already have the config used as well as the log)
    # results = {"history": history, "anythingelse": "here"}

    # # save the results as a python pickle
    # with open(str(results_path), "wb") as h:
    #     pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    with open('graphs.txt', 'a') as f:
        # f.write("Krum-full-flip_1-9(5) ")
        for _, second_element in history.metrics_centralized["accuracy"]:
            f.write(str(second_element) + ',')
        f.write("\n")
    



if __name__=="__main__":
    main()
