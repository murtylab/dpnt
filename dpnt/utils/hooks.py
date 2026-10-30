from functools import reduce

import torch


class ForwardHook:
    def __init__(self, model, hook_layer_name: str):
        layer = (
            reduce(getattr, hook_layer_name.split("."), model)
            if hook_layer_name
            else model
        )
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.input = None
        self.output = None
        self.hook_layer_name = hook_layer_name

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output.detach().clone() if torch.is_tensor(output) else output

    def close(self):
        self.hook.remove()

    def __repr__(self):
        return f"ForwardHook(hook_layer_name={self.hook_layer_name})"


def unit_axis(output: torch.Tensor) -> int:
    """the axis of a layer output that indexes its units: the channels of a conv map, the last dim
    otherwise. [N, H, W, C] (timm swin / samvit blocks) is a square token grid with the channels
    last; every other 4-D output is [N, C, H, W]. Detected by shape alone: this assumes square
    inputs and that no conv layer has as many channels as its map is wide (true for every model in
    the registry at 224)"""
    if output.dim() == 4:
        channels_last = (
            output.shape[1] == output.shape[2] and output.shape[3] != output.shape[2]
        )
        return 3 if channels_last else 1
    return output.dim() - 1


def pool_activation(output: torch.Tensor) -> torch.Tensor:
    """one layer's output as [N, D]: one value per unit per image"""
    if isinstance(output, (tuple, list)):
        output = output[0]
    if hasattr(output, "last_hidden_state"):  # HF ModelOutput
        output = output.last_hidden_state
    if output.dim() == 4:
        # conv maps and swin / samvit grids alike: average over space
        if unit_axis(output) == 3:
            return output.mean(dim=(1, 2))
        return output.mean(dim=(2, 3))
    if output.dim() == 3:
        # [N, S, D] transformer tokens (batch-first for every model in the
        # registry): average all S tokens, CLS included, whether or not the
        # model has a CLS token
        return output.mean(dim=1)
    return output  # already [N, D]
