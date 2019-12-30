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
The input of AttList includes the set of interactions between users and lists:

``userId [listId_1, listId_2, listId_3,...]``

which means that the user interacted with the lists (e.g., playlists on Spoftify). These interactions are stored in userList_\*.txt file, like AttList/AttList_cikm2019/data/goodreads/userList_goodreads.txt. 

Note that we split userList_\*.txt into train.txt, validation.txt and test.txt and only use train.txt in the training phase, otherwise the information will be leaked. Of course, validation.txt is for hyper-parameters selection and test.txt is for the evaluation.

The format of train.txt, validation.txt and test.txt are like this:

``userId listId 1.0``

And items contained in that list.

``listId [itemId_1, itemId_2, itemId_3,...]``

which means that the list contains these items. These containing relationships are storded in listItem_\*.txt, like AttList/AttList_cikm2019/data/goodreads/listItem_goodreads.txt

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
python Attlist_cikm2019.py --dataSetName Spotify

For more hyper-parameters, please see Attlist_cikm2019.py.

### Files in folder

#### attList_CIKM2019_data
- -goodreads.zip, unpreprocessed goodreads dataset;
- -spotify.zip, unpreprocessed spotify dataset;
- -zhihu.zip, unpreprocessed zhihu dataset;
- -examples_listContainItem.txt;
- -examples_userFollowingList.txt;

#### AttList_cikm2019
- -Attlist_cikm2019.py, the main function of AttList;
- -AttLayer_cikm2019.py, the file of vanilla attention module;
- -AttLayerSelf_cikm2019.py, the file of self-attention module;
- -getDataSet_cikm2019.py, the file to get data structure for training and testing;
- -evaluate_cikm2019.py, the file to do the evaluation;

#### Data (included in AttList_cikm2019)
- -goodreads: userList_goodreads.txt, listItem_goodreads.txt, train.txt, validation.txt and text.txt;
- -spotify: userList_spotify.txt, listItem_spotify.txt, train.txt, validation.txt and text.txt;
- -zhihu: userList_zhihu.txt, listItem_zhihu.txt, train.txt, validation.txt and text.txt;

### Citation
@inproceedings{he2019hierarchical,
  title={A Hierarchical Self-Attentive Model for Recommending User-Generated Item Lists},
  author={He, Yun and Wang, Jianling and Niu, Wei and Caverlee, James},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={1481--1490},
  year={2019},
  organization={ACM}
}

### Important Tip
The evaluation protocol in our paper is to generated the prediction scores for all lists in the dataset, which is slow. Hence, in the future, we may first randomly sample 100 negative samples. And the model only needs to predict scores for these 100 negative samples and ground-truth samples. This protocol has been widely used in recommendation research community.





