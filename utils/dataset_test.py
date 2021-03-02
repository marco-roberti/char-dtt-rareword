# coding=utf-8
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from datasets.base import DatasetType
from main import datasets, models
from utils.plot import show_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    which_set = DatasetType.TEST
    dataset = datasets[args.dataset](which_set)
    loader = DataLoader(dataset, shuffle=True, collate_fn=dataset.collate_fn)

    model = models[args.model](
        dataset.vocabulary_size(), dataset.sos_token, dataset.eos_token, dataset.pad_token).to(device)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        for x, y_ref in loader:
            x_str = dataset.to_string(x[0][0].tolist())
            y_ref_str = dataset.to_string(y_ref[0].tolist())

            y_prob, attention, p_gens = model(x)
            y_tensor = y_prob.argmax(1)
            y_list = y_tensor.tolist()
            y_str = dataset.to_string(y_list)
            attention = attention[:, 0].tolist()
            p_gens = [p_gens.squeeze().tolist()]

            print()
            print(f'> {x_str}')
            print(f'= {y_ref_str}')
            print(f'< {y_str}')
            show_attention(f'{x_str}', y_str, attention, p_gens)
            # show_attention_eda(x_str, y_str, attention)


if __name__ == '__main__':
    parser = ArgumentParser(description='Check a trained model on a dataset')
    parser.add_argument('weights', type=str, help='The state_dict of a trained model')
    main(parser.parse_args())
