import io
from typing import List
import torch
import torch.nn as nn
import math
import numpy as np
from aggregator import Aggregators
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor

def layerwise_model(model: torch.nn.Module, copy=False) -> List[torch.Tensor]:
    """Unfold model parameters by layer.
    
    Unfold every layer of model, concate tensor in the same layer, and
    return a list of torch.Tensor.

    Args:
        model (torch.nn.Module): model to unfold.
        copy (str): mode of model, True for copy, default is False.

    Returns:
        List[torch.Tensor]: list of unfolded parameters, [layer1, layer2, ...]
    """
    m_parameters = []

    for idx, (name, module) in enumerate(model.named_children()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if copy: 
                parameters = [param.clone().detach().view(-1) for param in module.parameters()]
            else:
                parameters = [param.detach().view(-1) for param in module.parameters()]
            param_cat = torch.cat(parameters)
            m_parameters.append(param_cat)
            # print(f"Layer {idx}: {name} - {param_cat.shape}")

    m_parameters = [param.cpu() for param in m_parameters]

    return m_parameters


def fg(timestamp, t, a):
    if a == 1:
        return 1
    return math.pow(a, -t + timestamp)


def gradient_diversity(gradients):
    norms = [torch.norm(grad, p=2, dim=0).item()**2 for grad in gradients]
    d = Aggregators.fedavg_aggregate(gradients)
    d_norm = torch.norm(d, p=2, dim=0).item()
    diversity = np.sqrt(sum(norms)/len(norms))/d_norm
    return diversity

def plot_to_tensorboard(x_data,y_data,tag,image,step,writer):
    """
    Plot the data to tensorboard.
    Args:
        x_data: np.array  
        y_data: List[List[float]] 
        tag: tag for the plot.
        image: image for the plot.
        step: step for the plot.
        writer: writer for the plot.
    """
    plt.figure(figsize=(8, 6))
    y_means = [np.mean(e) for e in y_data]
    y_stds = [np.std(e) for e in y_data]
    plt.plot(x_data, y_means)
    plt.fill_between(x_data, np.array(y_means) - np.array(y_stds), np.array(y_means) + np.array(y_stds), alpha=0.3)

    plt.xlabel("Communication Round")
    plt.ylabel("Local Update Norm")
    # plt.title("")
    plt.grid(True)

    # Convert the Matplotlib figure to a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = ToTensor()(image) # Convert PIL image to Tensor

    # Add the image to TensorBoard
    writer.add_image(tag, image, step)
    plt.close() # Close the figure to prevent display in notebook.