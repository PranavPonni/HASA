import torch
from torch import nn
import numpy as np
import os
import re
import pickle
import copy
from pytorch_msssim import ssim
import wandb
import pdb


def restore_data(path,run_id):
    parts = os.path.normpath(path).split(os.sep)
    front = os.sep.join(parts[:-2])  
    back = os.sep.join(parts[-2:])  
    wandb.restore(back,run_id,root=front)

def search_wandb_search(project,target_run_name,entity="takahisaueno"):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    target_run = None
    for run in runs:
        if run.name == target_run_name:
            target_run = run
            break

    if target_run:
        run_path = f"{target_run.entity}/{target_run.project}/{target_run.id}"
        print(f"Found run: {target_run.name}")
        print(f"Run path: {run_path}")
    else:
        print("Run not found")
        run_path=None

    return run_path

class LossScheduler:
    def __init__(self, decay_end=200, curve_name="s"):
        decay_start = 0
        self.counter = -1
        self.decay_end = decay_end
        self.interpolated_values = self.curve_interpolation(
            decay_start, decay_end, decay_end, curve_name
        )

    def linear_interpolation(self, start, end, num_points):
        x = np.linspace(start, end, num_points)
        return x

    def s_curve_interpolation(self, start, end, num_points):
        t = np.linspace(0, 1, num_points)
        x = start + (end - start) * (t - np.sin(2 * np.pi * t) / (2 * np.pi))
        return x

    def inverse_s_curve_interpolation(self, start, end, num_points):
        t = np.linspace(0, 1, num_points)
        x = start + (end - start) * (t + np.sin(2 * np.pi * t) / (2 * np.pi))
        return x

    def deceleration_curve_interpolation(self, start, end, num_points):
        t = np.linspace(0, 1, num_points)
        x = start + (end - start) * (1 - np.cos(np.pi * t / 2))
        return x

    def acceleration_curve_interpolation(self, start, end, num_points):
        t = np.linspace(0, 1, num_points)
        x = start + (end - start) * (np.sin(np.pi * t / 2))
        return x

    def curve_interpolation(self, start, end, num_points, curve_name):
        if curve_name == "linear":
            interpolated_values = self.linear_interpolation(start, end, num_points)
        elif curve_name == "s":
            interpolated_values = self.s_curve_interpolation(start, end, num_points)
        elif curve_name == "inverse_s":
            interpolated_values = self.inverse_s_curve_interpolation(
                start, end, num_points
            )
        elif curve_name == "deceleration":
            interpolated_values = self.deceleration_curve_interpolation(
                start, end, num_points
            )
        elif curve_name == "acceleration":
            interpolated_values = self.acceleration_curve_interpolation(
                start, end, num_points
            )
        else:
            assert False, "Invalid curve name. {}".format(curve_name)

        return interpolated_values / num_points

    def __call__(self, loss_weight):
        self.counter += 1
        if self.counter >= self.decay_end:
            return loss_weight
        else:
            return self.interpolated_values[self.counter] * loss_weight
        
def get_episode(directory):
    combined_dict = {}
    files = os.listdir(directory)
    files.sort(key=lambda x: int(x.split('timestep')[1].split('.pkl')[0]))
    timestep=0
    for file in files:
        timestep+=1
        if file.endswith('.pkl'):
            with open(os.path.join(directory, file), 'rb') as f:
                data = pickle.load(f)
                for key, value in data.items():
                    if key in combined_dict:
                        combined_dict[key]+=[value]
                    else:
                        combined_dict[key] = [value]

    for key,value in combined_dict.items():
        stacked=np.stack(value)
        if len(stacked.shape)==1:
            stacked=stacked[:,np.newaxis]
        combined_dict[key]=copy.deepcopy(stacked)

    return combined_dict


class TempScheduler:
    def __init__(self, start=1.0, end=0.0005, total_steps=1000, curve_name="s"):
        self.step_counter = -1
        self.total_steps = total_steps
        self.temperature_values = self.curve_interpolation(
            start, end, total_steps, curve_name
        )

    def linear_interpolation(self, start, end, num_points):
        x = np.linspace(start, end, num_points)
        return x

    def s_curve_interpolation(self, start, end, num_points):
        t = np.linspace(0, 1, num_points)
        x = start + (end - start) * (t - np.sin(2 * np.pi * t) / (2 * np.pi))
        return x

    def inverse_s_curve_interpolation(self, start, end, num_points):
        t = np.linspace(0, 1, num_points)
        x = start + (end - start) * (t + np.sin(2 * np.pi * t) / (2 * np.pi))
        return x

    def deceleration_curve_interpolation(self, start, end, num_points):
        t = np.linspace(0, 1, num_points)
        x = start + (end - start) * (1 - np.cos(np.pi * t / 2))
        return x

    def acceleration_curve_interpolation(self, start, end, num_points):
        t = np.linspace(0, 1, num_points)
        x = start + (end - start) * (np.sin(np.pi * t / 2))
        return x

    def curve_interpolation(self, start, end, num_points, curve_name):
        if curve_name == "linear":
            interpolated_values = self.linear_interpolation(start, end, num_points)
        elif curve_name == "s":
            interpolated_values = self.s_curve_interpolation(start, end, num_points)
        elif curve_name == "inverse_s":
            interpolated_values = self.inverse_s_curve_interpolation(start, end, num_points)
        elif curve_name == "deceleration":
            interpolated_values = self.deceleration_curve_interpolation(start, end, num_points)
        elif curve_name == "acceleration":
            interpolated_values = self.acceleration_curve_interpolation(start, end, num_points)
        else:
            assert False, "Invalid curve name. {}".format(curve_name)

        return interpolated_values

    def __call__(self):
        self.step_counter += 1
        if self.step_counter >= self.total_steps:
            return self.temperature_values[-1]
        else:
            return self.temperature_values[self.step_counter]


def get_activation_fn(name, inplace=True):
    if name.casefold() == "relu":
        return nn.ReLU(inplace=inplace)
    elif name.casefold() == "lrelu":
        return nn.LeakyReLU(inplace=inplace)
    elif name.casefold() == "softmax":
        return nn.Softmax()
    elif name.casefold() == "tanh":
        return nn.Tanh()
    elif name.casefold() == "mish":
        return nn.Mish()
    else:
        assert False, "Unknown activation function {}".format(name)


def l2_loss(input,teacher):
    l2_loss=torch.sum((input-teacher)**2,dim=1)

    return l2_loss

def gausian_noise(value,var,device=None):
    if device is not None:
        return torch.normal(mean=0.0,std=var, size=value.shape).to(device)
    else:
        return torch.normal(mean=0.0,std=var, size=value.shape)


def add_gausian_noise(value,var):
    device=value.device
    return value+gausian_noise(value,var,device)



def change_dir_name_in_path(path, old_dir_name, new_dir_name):
    path_parts = path.split(os.sep)
    path_parts = [new_dir_name if part == old_dir_name else part for part in path_parts]
    new_path = os.sep.join(path_parts)
    return new_path

def get_penultimate_dict(d):
    def _get_penultimate_dict(d, path=None):
        if path is None:
            path = []
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    yield from _get_penultimate_dict(v, path + [k])
                else:
                    yield path + [k], d

    result = {}
    for path, penultimate_dict in _get_penultimate_dict(d):
        if len(path) > 1:
            if path[-2] not in result:
                result[path[-2]] = {}
            result[path[-2]][path[-1]] = penultimate_dict[path[-1]]
    return result


def update_nested_dict(d, update_dict):
    for k, v in d.items():
        if isinstance(v, dict):
            update_nested_dict(v, update_dict)
        if k in update_dict:
            d[k] = update_dict[k]
    return d

def log_memory_usage(tag):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{tag}] Memory Usage: RSS={mem_info.rss / 1024 ** 2:.2f} MB, VMS={mem_info.vms / 1024 ** 2:.2f} MB")




def weight_init(model, initialization_dict):
    for name, module in model.named_parameters():
        print("Module:", name)
        if 'bias' in name:
            exec('nn.init.zeros_(module)')
            print(f"Initializing {name} using nn.init.zeros_")

        for init_name, init_func in initialization_dict.items():
            if re.fullmatch(init_name, name):
                if isinstance(init_func, (int, float)):
                    exec('nn.init.constant_(module,init_func)')
                    print(f"Initializing {name} using nn.init.constant_({init_func})")
                else:
                    print(f"Initializing {name} using {init_func}")
                    exec(f'{init_func}(module)')


def closed_loop(real, predict, rate):
    """
    This function returns a weighted sum of real and predict based on the given rate.

    Args:
    real: The real value (can be a scalar or an array).
    predict: The predicted value (can be a scalar or an array).
    rate: The rate at which real and predict should be weighted (should be a float between 0 and 1).

    Returns:
    Weighted sum of real and predict.
    """
    return real * rate + (1.0 - rate) * predict


def mask_mse_loss(pred, label, mask):
    """
    Compute the masked mean squared error loss.

    Parameters:
    pred (torch.Tensor): Predicted values, shape (batch, ...)
    label (torch.Tensor): Ground truth values, shape (batch, ...)
    mask (torch.Tensor): Mask tensor with 0.0 or 1.0, shape (batch, ...)

    Returns:
    torch.Tensor: The masked MSE loss.
    """
    mse_loss = nn.MSELoss(reduction='none')
    
    # Compute the MSE loss without reduction
    loss = mse_loss(pred, label)
    
    # Ensure the mask is broadcastable to the shape of loss
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    
    # Expand the mask to match the shape of the loss
    mask = mask.expand_as(loss)
    
    # Apply the mask to the loss
    masked_loss = loss * mask
    
    # Compute the mean over the masked elements
    masked_loss = masked_loss.sum() / mask.sum()
    
    return masked_loss

def mask_l1_loss(pred, label, mask):
    """
    Compute the masked L1 loss.

    Parameters:
    pred (torch.Tensor): Predicted values, shape (batch, ...)
    label (torch.Tensor): Ground truth values, shape (batch, ...)
    mask (torch.Tensor): Mask tensor with 0.0 or 1.0, shape (batch, ...)

    Returns:
    torch.Tensor: The masked L1 loss.
    """
    l1_loss = nn.L1Loss(reduction='none')
    
    # Compute the L1 loss without reduction
    loss = l1_loss(pred, label)
    
    # Ensure the mask is broadcastable to the shape of loss
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    
    # Expand the mask to match the shape of the loss
    mask = mask.expand_as(loss)
    
    # Apply the mask to the loss
    masked_loss = loss * mask
    
    # Compute the mean over the masked elements
    masked_loss = masked_loss.sum() / mask.sum()
    
    return masked_loss


def mask_ssim_loss(pred, label, mask):
    loss=ssim(pred,label,data_range=1,size_average=False)
    loss_average=torch.sum(mask.reshape(-1)*loss)/torch.sum(mask)
    return loss_average

def mask_cosine_loss(pred,mask):
    velocity = pred[:, 1:, :] - pred[:, :-1, :]
    position = pred[:, 1:, :] 
    inner_product = torch.sum(position * velocity, dim=-1)

    norm_position = torch.norm(position, dim=-1)
    norm_velocity = torch.norm(velocity, dim=-1)

    cosine_similarity = inner_product / (norm_position * norm_velocity + 1e-10)
    loss=torch.sum(cosine_similarity*mask[:,:-1])/mask.sum()
    return loss
