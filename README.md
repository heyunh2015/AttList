# AttList

### Introduction

This is the implementation of *AttList* as described in the paper:<br>
> A Hierarchical Self-Attentive Model for Recommending User-Generated Item Lists.<br>
> CIKM, 2019.<br>
> Yun He, Jianling Wang, Wei Niu and James Caverlee.<br>
> Department of Computer Science and Engineering, Texas A&M University.<br>
> Contact: yunhe@tamu.edu <br>
> Personal Page: http://people.tamu.edu/~yunhe/ <br>

The *AttList* algorithm recommends user-generated item lists (playlists on Spotify, book lists on Goodreads and answer collections on Zhihu) to other users.

### Usage 
AttList learns the user preference from training datasets and then ranks the lists for the users in testing datasets. Folder AttList_cikm2019 contains our code and preprocessed datasets with split used in our paper. Folder attList_CIKM2019_data contains raw datasets without preprocessed (which have been de-identified. No real user id, list id and item id are released, no worries!). 

#### Input
The input of AttList includes the set of interactions between users and lists from the training dataset:

``userId [listId_1, listId_2, listId_3,...]``

which means that the user interacted with the lists (e.g., playlists on Spoftify). These interactions are stored in userList_*.txt file, like AttList/AttList_cikm2019/data/goodreads/userList_goodreads.txt. Note that we split userList_*.txt into train.txt, validation.txt and test.txt and only use train.txt in the training phase, otherwise the information will be leaked. Of course, validation.txt is for hyper-parameters selection and test.txt is for the evaluation.

The format of train.txt, validation.txt and test.txt are like this:

``userId listId 1.0``

And items contained in that list.

``listId [itemId_1, itemId_2, itemId_3,...]``

which means that the list contains these items. These containing relationships are storded in listItem_*.txt, like AttList/AttList_cikm2019/data/goodreads/listItem_goodreads.txt

#### Output
The output are the evaluation results comparing the ranked lists of AttList and the groundtruth from the test.txt. An example is presented as follows:

```
precision@5: 0.0297358342335
recall@5: 0.0993438722178
precision@10: 0.0212903859373
recall@10: 0.139846678794
NDCG@5: 0.0738304666949
NDCG@10: 0.088376425288
```

#### Run
./PsiRec.py --train_file filePath --test_file filePath --walk_length 80 --num_walks 10

#### Parameters
- -train_file, the training dataset file;
- -test_file, the testing dataset file (or validation dataset file);
- -walk_length, the length of a random walk; the default is 80;
- -num_walks, the number of random walks visiting each user and item; the default is 10;
- -window_size, the context size for sampling the indirect user-item pairs; the default is 3;
- -user_number, the number of users in the dataset;
- -item_number, the number of items in the dataset;
- -train_epoch, the iterations of ALS matrix factorization; the default is 25;
- -lambda_value, the regularization value for ALS matrix factorization; the default is 0.25.
- -latent_factors, the number of latent factors of ALS matrix factorization; the default is 100;
- -validation, the Boolean variable to decide if do the validation on the validation dataset; the default is 0.

### Files in folder

#### PsiRec
- -PsiRec.py, the main function of PsiRec;
- -randomWalks.py, the file to generate random walks from the user-item bigraph;
- -SPPMI.py, the file to calculate Shifted Positive Pointwise Mutual Information (SPPMI) value as the confidence for each pseudo-implicit feedback;
- -alsMF.py, the file to apply latent factors model to predict the item lists for each user based on the dataset enriched by PsiRec;
- -evaludation.py, the file to do the evaluation;

#### Data
- -preProcessData.py, the file to preprocess the raw datasets, including transferring explicit datasets to implicit datasets and split the dataset into three parts: training, testing and validation;
- -DataInPaper, the datasets exactly the same in our paper, which can be used directly by PsiRec.py;
- -rawData, the raw datasets are too large to be handled here, but you can download according the URLs presented in our paper;
- -preProcessedData, the preprocessed datasets, which can be used directly by PsiRec.py;

### Citation
Pending.

### Acknowledgement
The technique of randomWalks.py is learned from https://github.com/aditya-grover/node2vec. Th first author is Aditya Grover from Standford University. Thanks to them!





