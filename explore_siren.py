#!/usr/bin/env python
# coding: utf-8

# # Siren Exploration
# 
# This is a colab to explore properties of the Siren MLP, proposed in our work [Implicit Neural Activations with Periodic Activation Functions](https://vsitzmann.github.io/siren).
# 
# 
# We will first implement a streamlined version of Siren for fast experimentation. This lacks the code to easily do baseline comparisons - please refer to the main code for that - but will greatly simplify the code!
# 
# **Make sure that you have enabled the GPU under Edit -> Notebook Settings!**
# 
# We will then reproduce the following results from the paper:
# * [Fitting an image](#section_1)
# * [Fitting an audio signal](#section_2)
# * [Solving Poisson's equation](#section_3)
# * [Initialization scheme & distribution of activations](#activations)
# * [Distribution of activations is shift-invariant](#shift_invariance)
# 
# We will also explore Siren's [behavior outside of the training range](#out_of_range).
# 
# Let's go! First, some imports, and a function to quickly generate coordinate grids.

# In[1]:


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time


import scipy.io.wavfile as wavfile
import io
from IPython.display import Audio

import argparse


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


# Now, we code up the sine layer, which will be the basic building block of SIREN. This is a much more concise implementation than the one in the main code, as here, we aren't concerned with the baseline comparisons.

# In[2]:


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


# And finally, differential operators that allow us to leverage autograd to compute gradients, the laplacian, etc.

# In[3]:


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


# ## Fitting an audio signal
# <a id='section_2'></a>
# 
# Here, we'll use Siren to parameterize an audio signal - i.e., we seek to parameterize an audio waverform $f(t)$  at time points $t$ by a SIREN $\Phi$.
# 
# That is we seek the function $\Phi$ such that:  $\mathcal{L}\int_\Omega \lVert \Phi(t) - f(t) \rVert \mathrm{d}t$  is minimized, in which  $\Omega$  is the domain of the waveform.
# 
# For the audio, we'll use the bach sonata:

# In[4]:




# if not os.path.exists('gt_bach.wav'):
#     get_ipython().system('wget https://vsitzmann.github.io/siren/img/audio/gt_bach.wav')


# Let's build a little dataset that computes coordinates for audio files:

# In[5]:


class AudioFile(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.rate, self.data = wavfile.read(filename)
        self.data = self.data.astype(np.float32)
        self.timepoints = get_mgrid(len(self.data), 1)

    def get_num_samples(self):
        return self.timepoints.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        amplitude = self.data
        scale = np.max(np.abs(amplitude))
        amplitude = (amplitude / scale)
        amplitude = torch.Tensor(amplitude).view(-1, 1)
        return self.timepoints, amplitude


# Let's instantiate the Siren. As this audio signal has a much higer spatial frequency on the range of -1 to 1, we increase the $\omega_0$ in the first layer of siren.

# In[42]:


# running on list of 5 audios

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_no', type = int, default = 1)
    parser.add_argument('--lr', type=float, default = 1e-4)
    parser.add_argument('--epochs', type=int, default = 1000)
    # parser.add_argument('--h', type=int, default = 512)

    args = parser.parse_args()

    file_no = args.file_no
    total_steps = args.epochs
    lr = args.lr

    file_1 = "307-127535-0021.wav"
    file_2 = "2514-149482-0013.wav"
    file_3 = "2514-149482-0063.wav"
    file_4 = "3240-131231-0044.wav"
    file_5 = "8238-274553-0062.wav"

    inp_audio_list = [file_1, file_2, file_3, file_4, file_5]

    file_path = "/vulcanscratch/ani01/838c_audio_deverb/input_wav" + "/" + inp_audio_list[file_no - 1]
    out_dir = "/vulcanscratch/ani01/838c_audio_deverb/output_wav/exp"

    print(f"input audio is: {file_path}")

    bach_audio = AudioFile(file_path)

    dataloader = DataLoader(bach_audio, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

    # Note that we increase the frequency of the first layer to match the higher frequencies of the
    # audio signal. Equivalently, we could also increase the range of the input coordinates.
    audio_siren = Siren(in_features=1, out_features=1, hidden_features=256,
                        hidden_layers=3, first_omega_0=3000, outermost_linear=True)
    audio_siren.cuda()


    # Let's have a quick listen to ground truth:

    rate, _ = wavfile.read(file_path)

    model_input, ground_truth = next(iter(dataloader))
    # Audio(ground_truth.squeeze().numpy(),rate=rate)

    # We now fit the Siren to this signal.

    # total_steps = 5000
    print(f"training epochs are: {total_steps}\n")

    steps_til_summary = 100

    # lr=3e-5
    print(f"learning rate is: {lr}\n")

    optim = torch.optim.Adam(lr=lr, params=audio_siren.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    loss_values = []

    for step in range(total_steps):
        model_output, coords = audio_siren(model_input)
        loss = F.mse_loss(model_output, ground_truth)

        loss_values.append(loss)

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))

            # fig, axes = plt.subplots(1,2)
            # axes[0].plot(coords.squeeze().detach().cpu().numpy(),model_output.squeeze().detach().cpu().numpy())
            # axes[1].plot(coords.squeeze().detach().cpu().numpy(),ground_truth.squeeze().detach().cpu().numpy())
            # plt.show()

        optim.zero_grad()
        loss.backward()
        optim.step()

    final_model_output, coords = audio_siren(model_input)
    audio_file_output = Audio(final_model_output.cpu().detach().squeeze().numpy(),rate=rate)


    with open(out_dir + '/' + file_path.split('/')[-1] + '_lr_' + str(lr) + '_epoch_' + str(total_steps), 'wb') as f:
        f.write(audio_file_output.data)


    mae = torch.mean(torch.abs(final_model_output - model_input)).item()
    print(f"mae is: {mae}\n")


    mse = torch.mean((final_model_output - model_input) ** 2).item()
    print(f"mse is: {mse}\n")

    fig, ax1 = plt.subplots()

    ax1.plot(range(1, total_steps + 1), loss_values)

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='g')

    plt.title(f'Loss plot | LR: {lr} | MAE: {mae} | MSE: {mse}')
    # plt.show()

    plt.savefig('/vulcanscratch/ani01/838c_audio_deverb/output_wav/plots' + '/' + str(file_no) + '_lr_' + str(lr) + '_epochs_' + str(total_steps) + '.png')





