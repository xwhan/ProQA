# ProQA

Resource-efficient method for pretraining a dense corpus index for open-domain QA and IR. Given a question, you could use this code to retrieval relevant paragraphs from Wikipedia and extract answers.

## 1. Set up the environments
```
conda create -n proqa -y python=3.6.9 && conda activate proqa
pip install -r requirements.txt
```
You need use follow [Nvidia Apex repo](https://github.com/NVIDIA/apex) to install Apex if you GPUs support fp16. 

## 2. Use pretrained index and models
Download the pretrained models and data from google drive:
```
pip install gdown
gdown https://drive.google.com/uc?id=1cJGbeIg6hekVytcVphZt8_EzubtK1IP- && unzip pretrained_models.zip
gdown https://drive.google.com/uc?id=1hfsfQHShvYsK0gbHItM4B1OONNbgtmrK && unzip data.zip
```
The data folder includes the QA datasets and also the paragraph database ``nq_paras.db`` which can be used with sqlite3. 

## 3. Retriever pretraining
### Pretraining with a single pretraining file:
* Pretraining under the "retrieval" directory: ``sh train_retriever_single.sh``

### Pretraining with clutered pretrianing files:
#### Generate clusters
* Generate the paragraph embeddings: ``sh get_para_embed.sh``
* Generate clusters using the paragraph embeddings: ``python group_paras.py``

#### Pretraining using clusters
* Then run the retrieval script: ``sh train_retriever_cluster.sh``

## 4. QA finetuning
* Finetune the pretraining model on the QA dataset under "qa" directory: ``train_dense_qa.sh``
