from torch import nn
import torch.nn.init as init

def initialize_model(model, initialization='xavier'):
    """
    Initialize Model Parameters

    Parameters:
    model (nn.Module): model being initialized
    initialization (str): initialization methods -> 'xavier','he','zeros'

    return:
    nn.Module: initialized model
    """
    if initialization == 'xavier':
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                init.xavier_normal_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
    elif initialization == 'he':
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)
    elif initialization == 'zeros':
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                init.zeros_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
    # Other initialization...
    return model