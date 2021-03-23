# The Rare Word Issue in Natural Language Generation: a Character-Based Solution
This repository contains the source code and the datasets used for the journal paper [_Rare Word Issue in Natural Language Generation: a Character-Based Solution_](https://www.mdpi.com/2227-9709/8/1/20/pdf) by [Giovanni Bonetta](https://www.linkedin.com/in/giovanni-bonetta-11276b105/), [Marco Roberti](http://www.di.unito.it/~roberti/), [Rossella Cancelliere](http://www.di.unito.it/~cancelli/), and [Patrick  Gallinari](https://www.lip6.fr/actualite/personnes-fiche.php?ident=P33).

## Step-by-step guide
### Requirements
Prior to use the code, install the following packages. Versions used in the experiments are reported; the code should work in more recent versions too.
* Python (3.7.7)
* tqdm (1.19.2)
* NumPy (4.56.0)
* PyTorch (1.7.1)
* Matplotlib (3.1.3)

### Training
The `main.py` file is used to train the `ED+A` or `ED+ACS` model. The only required argument is a random seed:
```bash
python3 main.py {seed} -d E2E -m EDA
python3 main.py {seed} -d E2E -m EDACS  # default
python3 main.py {seed} -d E2ENY -m EDA
python3 main.py {seed} -d E2ENY -m EDACS
```

Different hyperparameters can be set via argparse (run `python3 main.py -h` for more details):
```bash
  --dataset {E2E,E2ENY}  # default: E2E
  --model {EDA,EDACS}  # default: EDACS
  --attention_size 128
  --embedding_size 32
  --hidden_size 300
  --layers 3
  --total_epochs 32
  --batch_size 32
  --learning_rate 0.001
  --clip_norm CLIP_NORM 5
  --cosine_tmax 50000  # T_max argument for CosineAnnealingLR
  --cosine_etamin 0  # eta_min argument for CosineAnnealingLR
```

At the end of the training phase, one checkpoint for each epoch will be stored in the `trained_nets/{timestamp}/` folder, where `timestamp` is the UNIX time of starting the script.

### Generation
The `create_eval_files.py` script will generate both outputs and references files, which can be directly used as inputs for the evaluation script. For example, you can generate on the E2E development set using `ED+ACS` as follows:
```bash
PYTHONPATH=. python3 utils/create_eval_files.py {seed} trained_nets/{timestamp}/{checkpoint} dev -d E2E -m EDACS  # default
```
This will create the `trained_nets/{timestamp}/{checkpoint}.dev.output` and `trained_nets/{timestamp}/{checkpoint}.dev.references` files.

You can choose the dataset and the architecture via argparse. Different architecture's arguments used for training **must be set accordingly** (run `PYTHONPATH=. python3 utils/create_eval_files.py -h` for more details):
```bash
--dataset {E2E,E2ENY}  # default: E2E
--model {EDA,EDACS}  # default: EDACS
--attention_size 128
--embedding_size 32
--hidden_size 300
--layers 3
```

### Evaluation
We took advantage of the [E2E NLG Challenge Evaluation metrics](https://github.com/tuetschek/e2e-metrics). Please refer to their repository for detailed instructions.s

## Citations
Please use the following BibTeX snippet to cite our work:

```BibTeX
@Article{informatics8010020,
AUTHOR = {Bonetta, Giovanni and Roberti, Marco and Cancelliere, Rossella and Gallinari, Patrick},
TITLE = {The Rare Word Issue in Natural Language Generation: A Character-Based Solution},
JOURNAL = {Informatics},
VOLUME = {8},
YEAR = {2021},
NUMBER = {1},
ARTICLE-NUMBER = {20},
URL = {https://www.mdpi.com/2227-9709/8/1/20},
ISSN = {2227-9709},
DOI = {10.3390/informatics8010020}
}
```
