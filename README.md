DataMining
======
#### [Homework1——Vector Space Model + KNN](#homework1)
#### [Homework2——Naive Bayes Classifier](#homework2)
#### [Homework3——Clustering with sklearn](#homework3)

Homework1
------
## ——Vector Space Model + KNN
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
Homework2
------
## ——Naive Bayes Classifier
### Task
    Implement the naive bayes classifier to test its effect on the 20 Newsgroups dataset.
    
### Dataset
    The source of the original dataset is the same as homework1. The data of homework2 is the data after homework1 divides the training set and the test set. See homework1 for the specific code.

### Theory
多项式模型(multinomial model)即为词频型和伯努利模(Bernoulli model)即文档型。</br>
多项式模型(multinomial model)：</br>
![多项式](https://github.com/LCabbage/201834869LiZongbu/raw/master/Homework2/multinomialModel.png)  </br>
伯努利模型(Bernoulli model)：</br>
![伯努利](https://github.com/LCabbage/201834869LiZongbu/raw/master/Homework2/BernoulliModel.png) </br>
比较：</br>
![compare](https://github.com/LCabbage/201834869LiZongbu/raw/master/Homework2/compare.png) </br>
Here I use the naive bayes multinomial model for classification.
### Usage
* NaiveBayes.py:</br>
    1.Count the number of occurrences of each word in each category and the total number of all words in each category.</br>
    2.Calculate conditional probabilities and prior probabilities, use a multinomial model, and use smoothing techniques (to avoid zero) and logarithm (to speed up calculations).</br>
    3.Classify the text, select the one with the highest probability, and save the results to the document.</br>
* compute_acc.py</br>
    Read saved result documents (number, category, forecast category), divide it by spaces, compare whether the category and forecast category are the same, and then calculate the accuracy.
### Performance
    分类正确的个数： 3042
    测试集总文档数： 3772
    准确率： 0.8064687168610817
###        
Homework3
------
##  ——Clustering with sklearn
### Task
    1.Test the clustering effect of the following clustering algorithm in sklearn on tweets data set.
    2.The NMI(Normalized Mutual Information) is used as the evaluation index.
### Dataset
    The Tweets dataset is in format of JSON like follows:
    {"text": "centrepoint winter white gala london", "cluster": 65}
    {"text": "mourinho seek killer instinct", "cluster": 96}
    {"text": "roundup golden globe won seduced johansson voice", "cluster": 72}
    {"text": "travel disruption mount storm cold air sweep south florida", "cluster": 140}
### Requirements 
    Python (>=2.6 或 >=3.3 版本)（这里python3）
    Numpy (>=1.6.1)
    Scipy (>=0.9)
### Propaedeutics
    scikit-learn
    tf-idf
    clustering methods
    Clustering performance evaluation
    PCA
    matplotlib
   ![sklearn](https://github.com/LCabbage/201834869LiZongbu/blob/master/Homework3/learningNotes/scikit-learn.png)
### Realization process    
    1. Read the text content of the training set
    After reading each document in the folder of a given data set, the text content is written into a TXT, and each behavior is a document, which facilitates the processing of word frequency matrix later. 
    2. Text preprocessing
    Read the result.txt that holds all the text content before, remove Spaces, punctuation, and word segmentation with stutter. 
    (The above two steps, the teacher has achieved)
    3. Feature extraction
    Using the tool scikit-learn, CountVectorizer() and TfidfTransformer() are called to calculate the tf-idf value, and the text is converted into a word frequency matrix. The matrix element a[I][j] represents the word frequency of j words in class I text.Save the word frequency matrix in the tf-idf_result document. 
    4. Each clustering algorithm
    Set parameters according to the characteristics of the clustering algorithm, call sklear.cluster implementation, save the clustering model, and use it for the test set.
    (At the beginning, the implementation of each algorithm was written in a program, but in order to view it separately, each algorithm was written separately, so only the partial clustering method was dimension-reduced and visualized.)
    5. Test set classification
    Test the test set text using clf.fit_predict. 
    6. Clustering performance evaluation (used here is: NMI)
    8. Call PCA to lower the dimension
    7. Call matplotlib for visualization
### Usage
  * sklearn_tfidf.py</br>
    1.Read the raw data and save the text content and clustering categories separately for later use.</br>
    2.Use the scikit-learn tool to call the CountVectorizer() and TfidfTransformer() functions to calculate the TF-IDF value, and convert the text into a word frequency matrix. The matrix element a[i][j] represents the word frequency of the j word under the i-type text. Save the word frequency matrix in the tfidf_result.txt.
  * Running program for each clustering algorithm
### Performance
    The results of each algorithm have been saved to the document, see: result.txt.
    
#### [回到顶部](#datamining)
