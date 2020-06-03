# ProQA

Resource-efficient method for pretraining a dense corpus index for open-domain QA and IR. Given a question, you could use this code to retrieval relevant paragraphs from Wikipedia and extract answers. More details about this method can be found in our paper [https://arxiv.org/pdf/2005.00038.pdf](https://arxiv.org/pdf/2005.00038.pdf).

## 1. Set up the environments
```
conda create -n proqa -y python=3.6.9 && conda activate proqa
pip install -r requirements.txt
```
If you want to used mixed precision training, you need to follow [Nvidia Apex repo](https://github.com/NVIDIA/apex) to install Apex if your GPUs support fp16. 

## 2. Download data (including the corpus, paragraphs paired with the generated questions, etc.)
```
gdown https://drive.google.com/uc?id=1-9BKTa82wL_CXKtwSlD_2lqfZpEmSkLl && unzip proqa_data.zip -d /data
```
The data folder includes the QA datasets and also the paragraph database ``nq_paras.db`` which can be used with sqlite3. 

## 2. Use pretrained index and models
Download the pretrained models and data from google drive:
```
gdown https://drive.google.com/uc?id=1cJGbeIg6hekVytcVphZt8_EzubtK1IP- && unzip pretrained_models.zip
```

### Test the Retrieval Performance Before QA finetuning
* First, encode all the questions as embeddings (use WebQuestions text for this example):
```
cd retrieval
CUDA_VISIBLE_DEVICES=0 python get_embed.py \
    --do_predict \
    --predict_batch_size 512 \
    --bert_model_name bert-base-uncased \
    --efficient_eval \
    --predict_file ../data/WebQuestions-test.txt \
    --init_checkpoint ../pretrained_models/retriever.pt \
    --is_query_embed \
    --embed_save_path ../data/wq_test_query_embed.npy
```

* Retrieval topk (k=80) paragraphs from the corpus and evaluate recall with simple string matching
```
python eval_retrieval.py ../data/WebQuestions-test.txt ../pretrained_models/para_embed.npy ../data/wq_test_query_embed.npy ../data/wiki_paras.db
```
The arguments are the dataset file, dense corpus index, question embeddings and the paragraph database. The results should be like:
```
Top 80 Recall for 2032 QA pairs: 0.7568897637795275 ...
Top 5 Recall for 2032 QA pairs: 0.468996062992126 ...
Top 10 Recall for 2032 QA pairs: 0.5674212598425197 ...
Top 20 Recall for 2032 QA pairs: 0.6441929133858267 ...
Top 50 Recall for 2032 QA pairs: 0.7253937007874016 ...
```

### Retrieval Demo
TODO

### Test the Final QA Performance
TODO


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
    --predict_batch_size 2048 \
    --bert_model_name bert-base-uncased \
    --efficient_eval \
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
* Finetune the pretraining model on the QA dataset under "qa" directory: ``./train_dense_qa.sh``
