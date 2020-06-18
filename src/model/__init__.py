import torch

from utils import coerce_to_path_and_check_exist
from .dtigmm import DTIGMM
from .dtikmeans import DTIKmeans
from .tools import safe_model_state_dict


def get_model(name):
    return {
        'dtikmeans': DTIKmeans,
        'dtigmm': DTIGMM,
    }[name]


def load_model_from_path(model_path, dataset, device=None, attributes_to_return=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(coerce_to_path_and_check_exist(model_path), map_location=device.type)
    model = get_model(checkpoint['model_name'])(dataset, **checkpoint['model_kwargs'])
    model = model.to(device)
    model.load_state_dict(safe_model_state_dict(checkpoint['model_state']))
    if attributes_to_return is not None:
        if isinstance(attributes_to_return, str):
            attributes_to_return = [attributes_to_return]
        return model, [checkpoint.get(key) for key in attributes_to_return]
    else:
        return model
