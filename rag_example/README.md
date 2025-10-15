# RAG demo

The folder contains a simple application that uses RAG concept, namely: we provide an fictional knowledge base in form of files and then take advantage of a small LLM for searching in that knowledge base using word embedding of that LLM. To facilitate the word search we use the vectorDB Chroma that internally uses HNSW algorithm (Hierarchical Navigable Small World graph).

## Installation

Run the command to install the dependencies:

```sh
pip install -r requirements.txt
```

In case you want to use only CPU (instead of GPU) refer to the guide:

https://huggingface.co/docs/transformers/installation

Also don't forget to set env variables:

```sh
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_CACHE=... # path where do you want to store huggingface cache
```

## Running

After that you can launch the script with

```sh
python rag_demo.py
```

### Output

Once launched you will see the output:

```
Loading documents...
Loaded 32 document sections from /home/sasha/rag_example/knowledge_base
Building or loading vector index (HNSW)...
Creating RAG QA chain...
Device set to use cpu

answer_ag14: AG14 is a strong painkiller for animals. It is used by veterinarians to treat severe pain in livestock and pets. It's used to treat pain in pets and livestock. It can also be used as a painkiller in humans. It was developed in the 1950s and 1960s.

===================================

answer_btx4: BTX4 is an adaptive encryption algorithm for IoT devices. It automatically scales key length based on network threat levels detected in real-time. It is based on a series of algorithms that are designed to work with different types of devices. The algorithm is designed to scale key length according to the network threat level.

>>> TEST 2: Without vectorDB (no retrieval)
Device set to use cpu
LLM-only answer (no retrieval): Suggest an algorithm for encryption of IoT devices. Suggest an algorithm to make it easier for devices to be encrypted. Share your thoughts on how you would like to see this technology used in the future. Send your ideas to: jennifer.smith@mailonline.co.uk.

```
