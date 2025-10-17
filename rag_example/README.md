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
Loading documents...
Loaded 32 document sections from ./knowledge_base
Building or loading vector index (HNSW)...
Creating RAG QA chain...
llama_context: n_batch is less than GGML_KQ_MASK_PAD - increasing to 64
llama_context: n_ctx_per_seq (4096) < n_ctx_train (32768) -- the full capacity of the model will not be utilized

answer_ag14: 
AG14 is a strong painkiller for animals. It is used by veterinarians to treat severe pain in livestock and pets.

===================================

answer_btx4: 
BTX4 is a suitable encryption algorithm for IoT devices. It's an adaptive algorithm that automatically scales key length based on real-time network threat levels detection.

===================================

>>> TEST 2: Without vectorDB (no retrieval)
llama_context: n_batch is less than GGML_KQ_MASK_PAD - increasing to 64
llama_context: n_ctx_per_seq (4096) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
LLM-only answer (no retrieval): .

Designing an encryption algorithm for IoT devices involves considering the unique characteristics and constraints of these devices. Here's a suggested encryption algorithm for securing IoT devices:

1. Preprocessing: Generate a random key (K) and initialization vector (IV) of fixed size for each device. The IV should be unique for every message encrypted using the same key. Store the generated key, IV, and their corresponding MAC (Message Authentication Code) in a secure cloud storage or on-device memory.

2. Data Encryption: When data needs to be transmitted from an IoT device to a server or another device, the following steps are taken for encryption:

   a. Append a fixed-size MAC (Message Authentication Code) calculated using a secret key known only to the communicating parties. This ensures that the message has not been tampered with during transmission.

   b. Concatenate the IV, encrypted data, and MAC.

   c. Use a symmetric encryption algorithm such as AES (Advanced Encryption Standard) in CBC (Cipher Block Chaining) mode to encrypt the concatenated IV, data, and MAC. The output of this step is the ciphertext.

3. Data Decryption: When an IoT device or server receives encrypted data from another device or server, it follows these steps for decryption:

   a. Verify the received MAC against the MAC stored in the cloud storage or on-device memory to ensure that the message has not been tampered with during transmission.

   b. Extract the IV from the encrypted data.

   c. Use the symmetric encryption algorithm (AES) in CBC (Cipher Block Chaining) mode with the previously generated key K to decrypt the ciphertext. The output of this step is the plaintext data.

4. Data Transmission: Once the IoT device or server has decrypted the received data, it can be transmitted to the intended recipient for further processing. This may involve storing the data in a database, analyzing the data using machine learning algorithms, or sending the data to another IoT device or server for further processing.

5. Secure Storage and Retrieval: It is essential to ensure that the encrypted data is securely stored and retrieved when needed.

```
