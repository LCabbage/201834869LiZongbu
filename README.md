Data Mining
======
Homework1——Vector Space Model + KNN
------
### Task
* Preprocess the text data set and get the VSM representation of each text.
* Implement KNN classifier and test its effect on 20Newsgroups.
* 20% is used as the test data set to ensure the uniform distribution of documents of each category in the test data
### Dataset
  The [20 Newsgroups dataset(20news-18828.tar.gz)](http://qwone.com/~jason/20Newsgroups/
) is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. 

### Requirements
* python=3.6.5
* word segmentation :nltk

### Usage
  The five programs are marked in order, and executed in order, but after the second program (splitData.py), the training set and the test set need to be processed separately, so pay attention to modify the path in the two programs of featureFilter.py and tf_idf.py. This is also explained in the program.

### Performance
* k=15 accuracy : 0.830859
* k=20 accuracy : 0.837222
* k=25 accuracy : 0.836691

###
Homework2——Naive Bayes classifier
------
### Task
    Implement the naive bayes classifier to test its effect on the 20 Newsgroups dataset.
    
### Dataset
    The source of the original dataset is the same as homework1. The data of homework2 is the data after homework1 divides the training set and the test set. See homework1 for the specific code.

### Theory
多项式模型(multinomial model)即为词频型和伯努利模(Bernoulli model)即文档型。</br>
![多项式](https://github.com/LCabbage/201834869LiZongbu/raw/master/Homework2/multinomialModel.png)  </br>
![伯努利](https://github.com/LCabbage/201834869LiZongbu/raw/master/Homework2/BernoulliModel.png) </br>
![compare](https://github.com/LCabbage/201834869LiZongbu/raw/master/Homework2/compare.png) </br>
Here I use the naive bayes multinomial model for classification.
### Usage
* NaiveBayes.py:</br>
    Count the number of occurrences of each word in each category and the total number of all words in each category.</br>
    Calculate conditional probabilities and prior probabilities, use a multinomial model, and use smoothing techniques (to avoid zero) and logarithm (to speed up calculations).</br>
    Classify the text, select the one with the highest probability, and save the results to the document.</br>
* compute_acc.py</br>
    Read saved result documents (number, category, forecast category), divide by space, compare whether the category and forecast category are the same, and then calculate the accuracy.
### Performance
    分类正确的个数： 3042
    测试集总文档数： 3772
    准确率： 0.8064687168610817
    
    
