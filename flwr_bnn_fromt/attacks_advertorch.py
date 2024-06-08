import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.nn as nn
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH
filename = "mnist_lenet5_clntrained.pt"
# filename = "mnist_lenet5_advtrained.pt"

model = LeNet5()
model.load_state_dict(
torch.load(os.path.join(TRAINED_MODEL_PATH, filename),map_location=torch.device('cpu') ))
model.to(device)
model.eval()

batch_size = 5
loader = get_mnist_test_loader(batch_size=batch_size)
for cln_data, true_label in loader:
    break
cln_data, true_label = cln_data.to(device), true_label.to(device)

from advertorch.attacks import LinfPGDAttack
adversary = LinfPGDAttack(
model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
targeted=False)


adv_untargeted = adversary.perturb(cln_data, true_label)
target = torch.ones_like(true_label) * 3
adversary.targeted = True
adv_targeted = adversary.perturb(cln_data, target)
print((target))


pred_cln = predict_from_logits(model(cln_data))
pred_untargeted_adv = predict_from_logits(model(adv_untargeted))
pred_targeted_adv = predict_from_logits(model(adv_targeted))
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
for ii in range(batch_size):
    plt.subplot(3, batch_size, ii + 1)
    _imshow(cln_data[ii])
    plt.title("clean \n pred: {}".format(pred_cln[ii]))
    plt.subplot(3, batch_size, ii + 1 + batch_size)
    _imshow(adv_untargeted[ii])
    plt.title("untargeted \n adv \n pred: {}".format(
    pred_untargeted_adv[ii]))
    plt.subplot(3, batch_size, ii + 1 + batch_size * 2)
    _imshow(adv_targeted[ii])
    plt.title("targeted to 3 \n adv \n pred: {}".format(
    pred_targeted_adv[ii]))

plt.tight_layout()
plt.show()
