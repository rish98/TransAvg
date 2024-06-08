from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

# from flwr.server.strategy import trans_autoencoder

import torch
import flwr as fl


import warnings
warnings.filterwarnings("ignore")

from model_alexnet_full import Net, train, test, generate_unique_filename, train_lf, train_ga, train_bd


class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, trainloader, vallodaer, num_classes,client_id_num) -> None:
        super().__init__()


        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = vallodaer

        # a model that is randomly initialised at first
        self.model = Net(num_classes)
        self.client_id_num =client_id_num

        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # for batch in self.trainloader:
        #             _, labels = batch[0].to(self.device), batch[1].to(self.device)
        #             for idx in range (len(labels)):
        #                 if labels[idx]==1:
        #                     labels[idx]=9

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict(
        {
            k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
            for k, v in params_dict
        }
    )


        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # fetch elements in the config sent by the server. Note that having a config
        # sent by the server each time a client needs to participate is a simple but
        # powerful mechanism to adjust these hyperparameters during the FL process. For
        # example, maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customising what you pass to `on_fit_config_fn` when
        # defining your strategy.
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]
        curr_round =config['current_round']

        # a very standard looking optimiser
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        # do local training. This function is identical to what you might
        # have used before in non-FL projects. For more advance FL implementation
        # you might want to tweak it but overall, from a client perspective the "local
        # training" can be seen as a form of "centralised training" given a pre-trained
        # model (i.e. the model received from the server)

        # if self.client_id_num not in [1,2,3,4,5]:
        train(self.model, self.trainloader, optim, epochs, self.device)

        if self.client_id_num in [1,2,3,4,5]:
            # train(self.model, self.trainloader, optim, epochs, self.device)
        #     print(self.client_id_num,"ATTACK!")
        #     with open('attacks.txt', 'a') as f:
        #         f.write(str((self.client_id_num,curr_round))+" ")
            # noise_std = 0.4
            for param in self.model.parameters():
                param.data = torch.mul(param.data, -1)
                # noise = torch.randn_like(param.data) * noise_std
                # param.data += noise
        #     for batch in self.trainloader:
        #         _, labels = batch[0].to(self.device), batch[1].to(self.device)
        #         for idx in range (len(labels)):
        #             if labels[idx]==1:
        #                 labels[idx]=9
        #             elif labels[idx]==9:
        #                 labels[idx]=1
        
        # if curr_round in [25,30,35,40,50,70,80,90,95,98,99,100] and self.client_id_num in [1,2,3,4,5]:
        #     filename = generate_unique_filename("./model_params_trans/mal_neg_params")
        #     torch.save(self.model.fc4,filename)
        # elif curr_round in [25,30,35,40,50,70,80,90,95,98,99,100] and self.client_id_num not in [1,2,3,4,5]:
        #     filename = generate_unique_filename("./model_params_trans/clean_params")
        #     torch.save(self.model.fc4,filename)

        # if "hello"  in config:
            
            # print( config["hello"],"\n")
            # for name, param in self.model.state_dict().items():
            #     print(name, param.size())

            # for param in self.model.parameters():
                # print("before",param.data)
                # option 1 multiply all by -1
                # param.data = torch.mul(param.data, -1)
                
                # option 2 add random noise
                # noise_std = 0.1
                # noise = torch.randn_like(param.data) * noise_std
                # param.data += noise

                # option 3 targeted class 
                # from advertorch.attacks import LinfPGDAttack
                # adversary = LinfPGDAttack(
                # self.model, loss_fn=torch.nn.CrossEntropyLoss(), eps=0.15,
                # nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
                # targeted=True)
                # adv_targeted = adversary.perturb(cln_data, target)

                # option 4 label flip
                # for batch in self.trainloader:
                # _, labels = batch[0].to(self.device), batch[1].to(self.device)
                #     for idx in range (len(labels)):
                #         if labels[idx]==1:
                #             labels[idx]=9
                #         elif labels[idx]==9:
                #             labels[idx]=1


                # print("after",param.data)

                # break

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)

        # if curr_round>27:
        #     filename = generate_unique_filename("./model_params_r3")
        #     torch.save(self.model.fc4,filename)
        return self.get_parameters({}), len(self.trainloader), {"client_id_num":self.client_id_num,"last_layer":self.model.fc2.weight.flatten()}
            # "client_score":trans_ae.get_similarity_score(self.model.fc4),  add this to return transformer score [trans]

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {"accuracy": accuracy}


def generate_client_fn(trainloaders, valloaders, num_classes):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            vallodaer=valloaders[int(cid)],
            num_classes=num_classes,
            client_id_num=int(cid)
        )

    # return the function to spawn client
    return client_fn