Data Mining
======

Homework1——Vector Space Model + KNN
------
## task
* Preprocess the text data set and get the VSM representation of each text.
* Implement KNN classifier and test its effect on 20Newsgroups.
* 20% is used as the test data set to ensure the uniform distribution of documents of each category in the test data
## Dataset
* The [20 Newsgroups dataset(20news-18828.tar.gz)](http://qwone.com/~jason/20Newsgroups/
) is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. 

## requirements
* python=3.6.5
* word segmentation :nltk

## usage
  The five programs are marked in order, and executed in order, but after the second program (splitData.py), the training set and the test set need to be processed separately, so pay attention to modify the path in the two programs of featureFilter.py and tf_idf.py. This is also explained in the program.

## performance
* k=15 accuracy : 0.830859
* k=20 accuracy : 0.837222
* k=25 accuracy : 0.836691
