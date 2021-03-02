# coding=utf-8
import os

from datasets.base import DatasetType
from datasets.e2e.e2e import E2E


class E2ENY(E2E):
    _csv_for = {
        DatasetType.TRAIN: os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trainset_ny.csv'),
        DatasetType.DEV:   os.path.join(os.path.dirname(os.path.realpath(__file__)), 'devset_ny.csv'),
        DatasetType.TEST:  os.path.join(os.path.dirname(os.path.realpath(__file__)), 'testset_ny.csv')
    }
