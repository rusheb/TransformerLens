import numpy as np
import torch


def get_sample_from_dataset(sequences, nb_sample=3, print_len=10):
    rd_idx = np.random.randint(0, len(sequences), nb_sample)
    return "\n".join([str(sequences[k][:print_len]) + " ... " for k in rd_idx])


def print_gpu_mem(step_name=""):
    print(f"{step_name} ~ {np.round(torch.cuda.memory_allocated()/1e9, 2)} GB allocated on GPU.")


def get_corner(tensor, n=2):
    "Gets the length n hypercube at the top left corner of the tensor"
    # this takes tensor[:n, :n, ..., :n] with as many :n slices as dimensions
    index = (slice(0, n),) * tensor.ndim
    return tensor[index]


def to_numpy(tensor, flat=False):
    if not isinstance(tensor, torch.Tensor): # catches parameters too
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()


def gelu_new(input):
    "Implementation of GeLU used by GPT2 - subtly different from PyTorch's"
    return 0.5 * input * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
