import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from collections import Counter
import random

phoneme_mapping = {
    'p': 'p',
    'b': 'b',
    't': 't',
    'd': 'd',
    'k': 'k',
    'g': 'g',
    'dx': 'dx',
    'f': 'f',
    'v': 'v',
    'dh': 'dh',
    'th': 'th',
    's': 's',
    'z': 'z',
    'r': 'r',
    'w': 'w',
    'y': 'y',
    'jh': 'jh',
    'ch': 'ch',
    'iy': 'iy',
    'eh': 'eh',
    'ey': 'ey',
    'ae': 'ae',
    'aw': 'aw',
    'ay': 'ay',
    'oy': 'oy',
    'ow': 'ow',
    'uh': 'uh',
    'ah': 'ah',
    'ax': 'ah',
    'ax-h': 'ah',
    'aa': 'aa',
    'ao': 'aa',
    'er': 'er',
    'axr': 'er',
    'hh': 'hh',
    'hv': 'hh',
    'ih': 'ih',
    'ix': 'ih',
    'l': 'l',
    'el': 'l',
    'm': 'm',
    'em': 'm',
    'n': 'n',
    'en': 'n',
    'nx': 'n',
    'ng': 'ng',
    'eng': 'ng',
    'sh': 'sh',
    'zh': 'sh',
    'uw': 'uw',
    'ux': 'uw',
    'pcl': 'sil',
    'bcl': 'sil',
    'tcl': 'sil',
    'dcl': 'sil',
    'kcl': 'sil',
    'gcl': 'sil',
    'h#': 'sil',
    'pau': 'sil',
    'epi': 'sil'
}

manner_of_articulation = {
    'b': 'stop',
    'p': 'stop',
    'd': 'stop',
    't': 'stop',
    'g': 'stop',
    'k': 'stop',
    'f': 'fricative',
    'v': 'fricative',
    'dh': 'fricative',
    'th': 'fricative',
    's': 'fricative',
    'z': 'fricative',
    'm': 'nasal',
    'n': 'nasal',
    'ng': 'nasal',
    'l': 'glide',
    'r': 'glide',
    'w': 'glide',
    'y': 'glide',
}

place_of_articulation = {
    'p': 'bilabial',
    'b': 'bilabial',
    'm': 'bilabial',
    'f': 'labiodental',
    'v': 'labiodental',
    'w': 'labiodental',
    'th': 'dental',
    'dh': 'dental',
    't': 'alveolar',
    'd': 'alveolar',
    's': 'alveolar',
    'z': 'alveolar',
    'n': 'alveolar',
    'l': 'alveolar',
    'r': 'alveolar',
    'k': 'velar',
    'g': 'velar',
    'ng': 'velar',
    'y': 'palatal',
}

voicing = {
    'p': 'unvoiced',
    'b': 'voiced',
    'm': 'voiced',
    'f': 'unvoiced',
    'v': 'voiced',
    'w': 'voiced',
    'th': 'unvoiced',
    'dh': 'voiced',
    't': 'unvoiced',
    'd': 'voiced',
    's': 'unvoiced',
    'z': 'voiced',
    'n': 'voiced',
    'l': 'voiced',
    'r': 'voiced',
    'k': 'unvoiced',
    'g': 'voiced',
    'ng': 'voiced',
    'y': 'voiced'
}

vowels_frontness = {
    'ih': 'front',     # pink
    'iy': 'front',     # green
    'eh': 'front',     # red
    'ae': 'front',     # sand
    'aa': 'back',      # coffee
    'ah': 'central',   # but
    'ao': 'back',      # bought
    'uw': 'back',      # boot
    'uh': 'near-back', # book
    'ax': 'central',   # dust
    'ix': 'central',   # roses
    'ux': 'central',   # dude
    # 'ax-h': 'vowel',
    # 'oy': 'diphtong',
    # 'ow': 'diphtong',
    # 'aw': 'diphtong',
    # 'ey': 'diphtong',
    # 'axr': 'rhotic',
    # 'er': 'rhotic',
}

vowels_roundedness = {
    'ih': 'unrounded',            # pink
    'iy': 'unrounded',            # green
    'eh': 'unrounded',            # red
    'ae': 'unrounded',            # sand
    'aa': 'rounded',              # coffee
    'ah': 'unrounded',            # but
    'ao': 'rounded',              # bought
    'uw': 'rounded',              # boot
    'uh': 'rounded',              # book
    'ax': 'unrounded',            # dust
    'ix': 'unrounded',            # roses
    'ux': 'rounded',              # dude
    # 'ax-h': 'vowel',
    # 'oy': 'diphtong',
    # 'ow': 'diphtong',
    # 'aw': 'diphtong',
    # 'ey': 'diphtong',
    # 'axr': 'rhotic',
    # 'er': 'rhotic',
}

vowels_openness = {
    'ih': 'near-close',     # pink
    'iy': 'close',          # green
    'eh': 'open-mid',       # red
    'ae': 'near-open',      # sand
    'aa': 'open',           # coffee
    'ah': 'near-open',      # but
    'ao': 'open-mid',       # bought
    'uw': 'close',          # boot
    'uh': 'near-close',     # book
    'ax': 'mid',            # dust
    'ix': 'close',          # roses
    'ux': 'close',          # dude
    # 'ax-h': 'vowel',
    'oy': 'diphtong',
    'ow': 'diphtong',
    'aw': 'diphtong',
    'ey': 'diphtong',
    'axr': 'rhotic',
    'er': 'rhotic',
}

def set_seed(seed):
    """Set random seed."""
    if seed == -1:
        seed = random.randint(0, 1000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_batches(inputs, batch_size):
    '''
    Split list of TIMIT inputs in batches of size batch_size.
    :param inputs: list of TIMIT instances (i.e. timit['train'] or timit['test'])
    :param batch_size: number of instances per batch
    :return: list of TIMIT instances split in batches
    '''

    # Extract the audio arrays from the input
    audio_arrays = [input["audio"]["array"] for input in inputs]

    # Calculate number of batches
    num_batches = math.ceil(len(audio_arrays) / batch_size)

    # Split list of audio arrays in batches
    input_batches = [audio_arrays[batch_size * y:batch_size * (y + 1)] for y in range(num_batches)]

    print(f"Number of batches of size {batch_size}:", len(input_batches))

    return input_batches

def balance_classes(X_instances, y_labels, averaged=False):

    balanced_data_X = []
    balanced_data_y = []

    class_distribution = Counter(y_labels)
    print('class distribution', class_distribution)
    num_instances_per_class = min(class_distribution.values())

    for label in class_distribution.keys():

        i = 1
        instances = []
        labels = []

        for x, y in zip(X_instances, y_labels):
            if y == label:
                instances.append(x)
                labels.append(y)
                if i == num_instances_per_class:
                    balanced_data_X.extend(instances)
                    balanced_data_y.extend(labels)
                    break
                i += 1

    balanced_class_distribution = Counter(balanced_data_y)
    print('balanced class distribution', balanced_class_distribution)

    return balanced_data_X, balanced_data_y


def tensorify(lst):
    """
    List must be nested list of tensors (with no varying lengths within a dimension).
    Nested list of nested lengths [D1, D2, ... DN] -> tensor([D1, D2, ..., DN)

    :return: nested list D
    """
    # base case, if the current list is not nested anymore, make it into tensor
    if type(lst[0]) != list:
        if type(lst) == torch.Tensor:
            return lst
        elif type(lst[0]) == torch.Tensor:
            return torch.stack(lst, dim=0)
        else:  # if the elements of lst are floats or something like that
            return torch.tensor(lst)
    current_dimension_i = len(lst)
    for d_i in range(current_dimension_i):
        tensor = tensorify(lst[d_i])
        lst[d_i] = tensor
    # end of loop lst[d_i] = tensor([D_i, ... D_0])
    tensor_lst = torch.stack(lst, dim=0)
    return tensor_lst


def data_loader(embedding_dict, layer_idx, averaged=False, l0=False, target_phonemes=None):
    X = []
    y = []

    # When working with averaged phoneme representations
    if averaged:
        for phoneme, embed in embedding_dict[layer_idx].items():
            if embed != None and phoneme != 'sil':
                if target_phonemes == None:
                    X.append(np.array(embed))
                    y.append(phoneme)
                else:
                    if phoneme in target_phonemes:
                        X.append(np.array(embed))
                        y.append(phoneme)

    # When working with frame representations
    else:
        for phoneme, embed_list in embedding_dict[layer_idx].items():
            for i in embed_list:
                if i != None and phoneme != 'q':
                    if target_phonemes == None:
                        # Use tensors when training L0 probes
                        if l0:
                            X.append(i.clone().detach())
                        # Use numpy arrays when training Logistic Regression from sklearn
                        else:
                            X.append(np.array(i))
                        y.append(phoneme_mapping[phoneme])
                    else:
                        if phoneme in target_phonemes:
                            # Use tensors when training L0 probes
                            if l0:
                                X.append(i.clone().detach())
                            # Use numpy arrays when training Logistic Regression from sklearn
                            else:
                                X.append(np.array(i))
                            y.append(phoneme)

    return X, y
