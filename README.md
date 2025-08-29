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
# on linux also install tkinter
# sudo apt-get install python3-tk
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
# on linux also install tkinter
# sudo apt-get install python3-tk
```

2. launch the script

```bash
python pytorch_nn_example.py
```

### lightning_classification.py

A simple example of neural network that solves a classification problem. In this example the data are generated in form of triplets (x, y, z) which are supplied to a function that calculates a value. After that the value is classified based on range it lies within (e.x. value in 0..10 -> 0 class, value in 11..20 -> 1 class, value in 21..30 -> 2 class, etc). This behavior is managed by the "DataGenerator" class.

How to launch:

1. install pytorch-lighting

```bash
pip install torch pytorch-lightning matplotlib
```

2. launch the script

```bash
python lightning_classification.py
```

### launch_llm_example.py & launch_lang_chain.py

A couple of examples of running LLMs from hugging face directly and also using the lang chain framework (LlamaCpp).
It's possible to launch them without GPU, thus on CPU. 

Before running the scripts you should set the environment variables below:

```
export CUDA_VISIBLE_DEVICES=""  # to deactivate GPU
export HF_HUB_CACHE=<your_local_path>/.cache/huggingface/hub # location where you want to store cache data for LLM (must have ~10GB)
export HUGGING_FACE_API_KEY=abc # your hugging face token
```

You can also create a temporary environment for the scripts since a lot of dependencies required. Below are cheatsheet commands:

```bash
python3 -m venv .venv # initiate a temporary environment venv
source .venv/bin/activate # activate temporary environment venv
deactivate # deactivate temporary environment venv
# to clean up data
pip freeze | xargs pip uninstall -y
pip cache purge
find ~ -type f -printf "%s %p\n" | sort -nr | head -20
```

Useful links:

https://huggingface.co/docs/transformers/installation

List of dependencies used:

```bash
accelerate==1.10.0
aiohappyeyeballs==2.6.1
aiohttp==3.12.15
aiosignal==1.4.0
annotated-types==0.7.0
anyio==4.10.0
attrs==25.3.0
certifi==2025.8.3
charset-normalizer==3.4.3
contourpy==1.3.3
cycler==0.12.1
dataclasses-json==0.6.7
diskcache==5.6.3
filelock==3.19.1
fonttools==4.59.1
frozenlist==1.7.0
fsspec==2025.7.0
greenlet==3.2.4
h11==0.16.0
hf-xet==1.1.8
httpcore==1.0.9
httpx==0.28.1
httpx-sse==0.4.1
huggingface-hub==0.34.4
idna==3.10
Jinja2==3.1.6
jsonpatch==1.33
jsonpointer==3.0.0
kiwisolver==1.4.9
langchain==0.3.27
langchain-community==0.3.29
langchain-core==0.3.75
langchain-text-splitters==0.3.9
langsmith==0.4.19
lightning-utilities==0.15.2
llama_cpp_python==0.3.16
MarkupSafe==3.0.2
marshmallow==3.26.1
matplotlib==3.10.5
mpmath==1.3.0
multidict==6.6.4
mypy_extensions==1.1.0
networkx==3.5
numpy==2.3.2
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.3
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.8.90
orjson==3.11.3
packaging==25.0
pillow==11.3.0
propcache==0.3.2
psutil==7.0.0
pydantic==2.11.7
pydantic-settings==2.10.1
pydantic_core==2.33.2
pyparsing==3.2.3
PyQt5==5.15.11
PyQt5-Qt5==5.15.17
PyQt5_sip==12.17.0
python-dateutil==2.9.0.post0
python-dotenv==1.1.1
pytorch-lightning==2.5.3
PyYAML==6.0.2
regex==2025.7.34
requests==2.32.5
requests-toolbelt==1.0.0
safetensors==0.6.2
sentencepiece==0.2.1
setuptools==80.9.0
six==1.17.0
sniffio==1.3.1
SQLAlchemy==2.0.43
sympy==1.14.0
tenacity==9.1.2
tokenizers==0.21.4
torch==2.8.0
torchmetrics==1.8.1
tqdm==4.67.1
transformers==4.55.4
triton==3.4.0
typing-inspect==0.9.0
typing-inspection==0.4.1
typing_extensions==4.14.1
urllib3==2.5.0
yarl==1.20.1
zstandard==0.24.0
```
