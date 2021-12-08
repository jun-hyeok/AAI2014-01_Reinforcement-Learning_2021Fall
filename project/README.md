# Project

## Requirements

Python 3.8+

```bash
pip install mlagents==0.22.0
pip install tensorflow
pip install numpy
pip install pandas
```

or

```bash
pip install -r requirements.txt
```

## File Structure

```bash
project/
├── envs/
│   ├── 1.0/
│   ...
├── examples/
│   ├── config.csv
│   ├── dddqn.py
│   ├── double_dqn.py
│   ├── dqn.py
│   ├── dueling_dqn.py
│   ...
├── logs/
├── saves/
├── utils/
├── README.md
├── agent.py
├── environment.py
├── learner.py
├── q_network.py
├── q_network_prev.py
└── requirements.txt

24 directories, 152 files
```

## Run TensorBoard

```bash
tensorboard --logdir [path]
```

e.g.

```bash
tensorboard --logdir logs/
```
