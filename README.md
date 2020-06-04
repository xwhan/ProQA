# ProQA

Resource-efficient method for pretraining a dense corpus index for open-domain QA and IR. Given a question, you could use this code to retrieval relevant paragraphs from Wikipedia and extract answers.

## 1. Set up the environments
```
conda create -n proqa -y python=3.6.9 && conda activate proqa
pip install -r requirements.txt
```
If you want to used mixed precision training, you need to follow [Nvidia Apex repo](https://github.com/NVIDIA/apex) to install Apex if your GPUs support fp16. 

## 2. Download data (including the corpus, paragraphs paired with the generated questions, etc.)
```
gdown https://drive.google.com/uc?id=17IMQ5zzfkCNsTZNJqZI5KveoIsaG2ZDt && unzip data.zip
cd data && gdown https://drive.google.com/uc?id=1T1SntmAZxJ6QfNBN39KbAHcMw0JR5MwL
```
The data folder includes the QA datasets and also the paragraph database ``nq_paras.db`` which can be used with sqlite3. 

## 2. Use pretrained index and models
Download the pretrained models and data from google drive:
```
gdown https://drive.google.com/uc?id=1fDRHsLk5emLqHSMkkoockoHjRSOEBaZw && unzip pretrained_models.zip
```

### Test the Retrieval Performance Before QA finetuning
* First, encode all the questions as embeddings (use WebQuestions text for this example):
```
cd retrieval
CUDA_VISIBLE_DEVICES=0 python get_embed.py \
    --do_predict \
    --predict_batch_size 512 \
    --bert_model_name bert-base-uncased \
    --fp16 \
    --predict_file ../data/WebQuestions-test.txt \
    --init_checkpoint ../pretrained_models/retriever.pt \
    --is_query_embed \
    --embed_save_path ../data/wq_test_query_embed.npy
```

* Retrieval topk (k=80) paragraphs from the corpus and evaluate recall with simple string matching
```
python eval_retrieval.py ../data/WebQuestions-test.txt ../pretrained_models/para_embed.npy ../data/wq_test_query_embed.npy ../data/nq_paras.db
```
The arguments are the dataset file, dense corpus index, question embeddings and the paragraph database. The results should be like:
```
Top 80 Recall for 2032 QA pairs: 0.7568897637795275 ...
Top 5 Recall for 2032 QA pairs: 0.468503937007874 ...
Top 10 Recall for 2032 QA pairs: 0.5679133858267716 ...
Top 20 Recall for 2032 QA pairs: 0.6441929133858267 ...
Top 50 Recall for 2032 QA pairs: 0.7263779527559056 ...
```

## 3. Retriever pretraining
### Use a single pretraining file:
* Under the `retrieval` directory: 
```
cd retrieval
./train_retriever_single.sh
```
This script will use the unclustered the data for pretraining. After certain updates, we will pause the training and use the following steps to cluster the data and continue training. This will save a checkpoint under `retrieval/logs/`.

### Use clutered data for pretraining:
#### Generate paragraph clusters
* Generate the paragraph embeddings using the checkpoint from last step: 
```
mkdir encodings
CUDA_VISIBLE_DEVICES=0 python get_embed.py --do_predict --prefix eval-para \
    --predict_batch_size 300 \
    --bert_model_name bert-base-uncased \
    --fp16 \
    --predict_file ../data/retrieve_train.txt \
    --init_checkpoint ../pretrained_models/retriever.pt \
    --embed_save_path encodings/train_para_embed.npy \
    --eval-workers 32 \
    --fp16
```
* Generate clusters using the paragraph embeddings: 
```
python group_paras.py
```
Clustering hyperparameter settings such as num of clusters can be found in `group_paras.py`.

#### Pretraining using clusters
* Then run the retrieval script: 
```
./train_retriever_cluster.sh
```

## 4. QA finetuning
* Generate the paragraph dense index under "retrieval" directory: ``./get_para_embed.sh``
* Finetune the pretraining model on the QA dataset under "qa" directory: ``./train_dense_qa.sh``
