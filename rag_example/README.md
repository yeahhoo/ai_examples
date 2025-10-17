# RAG demo

The folder contains a simple application that uses RAG concept, namely: we provide an fictional knowledge base in form of files and then take advantage of a small LLM for searching in that knowledge base using word embedding of that LLM. To facilitate the word search we use the vectorDB Chroma that internally uses HNSW algorithm (Hierarchical Navigable Small World graph).

## Installation

Run the command to install the dependencies:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
pip install -r requirements.txt
```

In case you want to use only CPU (instead of GPU) refer to the guide:

https://huggingface.co/docs/transformers/installation

After that download a model, for example mistral-7b-instruct-v0.2.Q4_K_M.gguf:

```sh
export MODEL_PATH=/home/sasha/models/mistral # define your path
mkdir -p $MODEL_PATH
cd $MODEL_PATH

curl -L -o mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

## Running

After that you can launch the script with

```sh
python rag_demo.py
```

### Output

Once launched you will see the output:

```
answer_ag14: 
AG14 is a strong painkiller for animals. It is used by veterinarians to treat severe pain in livestock and pets.

===================================

answer_btx4: 
BTX4 is a suitable encryption algorithm for IoT devices. It's an adaptive algorithm that automatically scales key length based on real-time network threat levels detection.

===================================

>>> TEST 2: Without vectorDB (no retrieval)

AG14 is a gene that encodes for a protein called AtAG14. This protein plays an essential role in the regulation of plant growth and development, particularly under stress conditions. The AG14 gene has been identified as a key regulator of various plant processes, including seed germination, root growth, shoot elongation, and photosynthesis. Additionally, AG14 has been shown to play a critical role in the response of plants to various environmental stresses, such as drought, salinity, temperature extremes, and heavy metal contamination. Overall, AG14 is an important gene that plays a crucial role in the regulation of plant growth and development under normal and stress conditions.

```
