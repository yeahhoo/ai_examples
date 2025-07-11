# AI examples for learning

The repo contains some scripts for learning AI.

Python version used for writing the scripts is **Python 3.13.5**.

### lightning_func_prediction.py

The file contains script for predicting a value of given function. You pass a function and some parameters such as: epochs, learning_rate, min_x, max_x and the script tries to predict a value in the interval from *min_x* to *max_x*. 
Be careful with periodic functions such as sin, cos, tan - in this case you must also pass unwrapped values for these functions as *features*.

How to launch:

1. install pytorch-lighting

```bash
pip install torch pytorch-lightning matplotlib
```

2. launch the script

```bash
python lightning_func_prediction.py
```

### pytorch_nn_example.py

A simple neural network that is manually assembled with declaring all the parameters. It just tries to build a graph based on data points defined in *Dataset*.

How to launch:

1. install pytorch-lighting

```bash
pip install torch pytorch-lightning matplotlib
```

2. launch the script

```bash
python pytorch_nn_example.py
```