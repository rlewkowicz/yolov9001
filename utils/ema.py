from copy import deepcopy
import math

def de_parallel(model):
    """De-parallelize a model, returning a single-GPU model."""
    return model.module if hasattr(model, 'module') else model

class ModelEMA:
    """Updated Exponential Moving Average (EMA) implementation.
    
    Keeps a moving average of everything in the model state_dict (parameters and buffers).
    For EMA details see References.
    
    To disable EMA set the `enabled` attribute to `False`.
    
    Attributes:
        ema (nn.Module): Copy of the model in evaluation mode.
        updates (int): Number of EMA updates.
        decay (function): Decay function that determines the EMA weight.
        enabled (bool): Whether EMA is enabled.
    
    References:
        - https://github.com/rwightman/pytorch-image-models
        - https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Initialize EMA for 'model' with given arguments.
        
        Args:
            model (nn.Module): Model to create EMA for.
            decay (float, optional): Maximum EMA decay rate.
            tau (int, optional): EMA decay time constant.
            updates (int, optional): Initial number of updates.
        """
        self.ema = deepcopy(de_parallel(model)).eval().float()
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (
            1 - math.exp(-x / tau)
        )  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters.
        
        Args:
            model (nn.Module): Model to update EMA from.
        """
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            unwrapped_model = getattr(model, '_orig_mod', model)
            msd = de_parallel(unwrapped_model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
