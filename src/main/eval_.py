"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import math

import torch
import yaml
from PIL import Image
from termcolor import colored
from utils.common_config import get_val_dataset, get_val_transformations, get_val_dataloader, \
    get_model
from utils.evaluate_utils import get_predictions
from utils.memory import MemoryBank
from utils.utils import fill_memory_bank

CONFIG_EXP = "configs/selflabel/selflabel_imagenet_200.yml"
MODEL = "/home/ITRANSITION.CORP/i.sechko/Downloads/selflabel_imagenet_200.pth.tar"


def visualize_indices(indices, dataset, hungarian_match):
    import matplotlib.pyplot as plt
    import numpy as np

    for idx in indices:
        img = np.array(dataset.get_image(idx)).astype(np.uint8)
        img = Image.fromarray(img)
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.show()


def visualize_indices_subplots(indices, dataset, class_number=0):
    import matplotlib.pyplot as plt
    import numpy as np

    if type(indices) == int or len(indices) == 1:
        fig, ax = plt.subplots(111)
        img_id = indices if type(indices) == int else indices[0]
        img = np.array(dataset.get_image(img_id)).astype(np.uint8)
        img = Image.fromarray(img)

        ax.imshow(img)
        ax.set_title(f"{img_id}", fontdict={"fontweight": 800})
        ax.axis("off")
        plt.show()
    else:
        side = math.ceil(math.sqrt(len(indices)))
        side = 10 if side > 10 else side
        fig, ax = plt.subplots(side, side)
        for idx, img_id in enumerate(indices):
            img = np.array(dataset.get_image(img_id)).astype(np.uint8)
            img = Image.fromarray(img)

            ax[idx // side, idx % side].imshow(img)
            ax[idx // side, idx % side].axis("off")

        fig.tight_layout(w_pad=0.1, h_pad=0.1)
        plt.show()
        # plt.savefig(fname=f"visualized/test{class_number}.png")


def evaluate(config_path, model_path):
    # Read config file
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    config['batch_size'] = 96  # To make sure we can evaluate on a single 1080ti
    print(config)

    # Get dataset
    print(colored('Get validation dataset ...', 'blue'))
    transforms = get_val_transformations(config)
    dataset = get_val_dataset(config, transforms)
    dataloader = get_val_dataloader(config, dataset)
    print('Number of samples: {}'.format(len(dataset)))

    # Get model
    print(colored('Get model ...', 'blue'))
    model = get_model(config)
    print(model)

    # Read model weights
    print(colored('Load model weights ...', 'blue'))
    state_dict = torch.load(model_path, map_location='cpu')

    if config['setup'] in ['simclr', 'moco', 'selflabel']:
        model.load_state_dict(state_dict)

    elif config['setup'] == 'scan':
        model.load_state_dict(state_dict['model'])

    else:
        raise NotImplementedError

    # CUDA
    model.cuda()

    # Perform evaluation
    if config['setup'] in ['simclr', 'moco']:
        print(colored('Perform evaluation of the pretext task (setup={}).'.format(config['setup']), 'blue'))
        print('Create Memory Bank')
        if config['setup'] == 'simclr':  # Mine neighbors after MLP
            memory_bank = MemoryBank(len(dataset), config['model_kwargs']['features_dim'],
                                     config['num_classes'], config['criterion_kwargs']['temperature'])

        else:  # Mine neighbors before MLP
            memory_bank = MemoryBank(len(dataset), config['model_kwargs']['features_dim'],
                                     config['num_classes'], config['temperature'])
        memory_bank.cuda()

        print('Fill Memory Bank')
        fill_memory_bank(dataloader, model, memory_bank)

        print('Mine the nearest neighbors')
        for topk in [1, 5, 20]:  # Similar to Fig 2 in paper
            _, acc = memory_bank.mine_nearest_neighbors(topk)
            print('Accuracy of top-{} nearest neighbors on validation set is {:.2f}'.format(topk, 100 * acc))


    elif config['setup'] in ['scan', 'selflabel']:
        print(colored('Perform evaluation of the clustering model (setup={}).'.format(config['setup']), 'blue'))
        head = state_dict['head'] if config['setup'] == 'scan' else 0
        predictions, features = get_predictions(config, dataloader, model, return_features=True)

        return predictions, features

    else:
        raise NotImplementedError


if __name__ == "__main__":
    preds, features = evaluate(CONFIG_EXP, MODEL)
    torch.save((preds, features), "tensors/selflabel_imagenet_200_footwear_output.pt")
