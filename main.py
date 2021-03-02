# coding=utf-8
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np

from datasets.base import DatasetType
from datasets.e2e.e2e import E2E
from datasets.e2e.e2e_newyork import E2ENY
from models.defaults import default_gru, default_embedding, default_attention
from models.eda import EDA
from models.eda_c import EDA_C
from utils.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets = {
    'E2E': E2E,
    'E2ENY': E2ENY
}

models = {
    'EDA': EDA,
    'EDA_C': EDA_C
}


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = datasets[args.dataset](DatasetType.TRAIN)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=True)

    model = models[args.model](dataset.vocabulary_size(), dataset.sos_token, dataset.eos_token, dataset.pad_token,
                               attention_size=args.attention_size, embedding_size=args.embedding_size,
                               hidden_size=args.hidden_size, num_layers=args.layers)

    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, args.cosine_tmax, args.cosine_etamin) if args.cosine_tmax else None
    criterion = NLLLoss()

    losses = train(data_loader, model, optimizer, scheduler, criterion, dataset.vocabulary_size(), args.n_epochs,
                   args.epoch, clip_norm=args.clip_norm)
    print(losses)


if __name__ == '__main__':
    parser = ArgumentParser(description='Utility script to (load and) train a model.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('seed', type=int)
    parser.add_argument('-d', '--dataset', type=str, default='E2E', choices=datasets.keys(), help=' ')
    parser.add_argument('-m', '--model', type=str, default='EDA_C', choices=models.keys(), help=' ')

    # Model parameters
    parser.add_argument('-a', '--attention_size', type=int, default=default_attention['size'], help=' ')
    parser.add_argument('-emb', '--embedding_size', type=int, default=default_embedding['size'], help=' ')
    parser.add_argument('-s', '--hidden_size', type=int, default=default_gru['hidden_size'], help=' ')
    parser.add_argument('-l', '--layers', type=int, default=default_gru['num_layers'], help=' ')

    # Training parameters
    parser.add_argument('-etot', '--total_epochs', dest='n_epochs', default=20, type=int, help=' ')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help=' ')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help=' ')
    parser.add_argument('--clip_norm', type=float, default=1, help=' ')
    parser.add_argument('--cosine_tmax', type=int, default=0, help='T_max argument for CosineAnnealingLR')
    parser.add_argument('--cosine_etamin', type=int, default=0, help='eta_min argument for CosineAnnealingLR')

    # Save/resume training
    parser.add_argument('-w', '--weights', help='The state_dict of a trained model', type=str)
    parser.add_argument('-e', '--epoch', help='The epoch index to start with', default=0, type=int)

    main(parser.parse_args())
