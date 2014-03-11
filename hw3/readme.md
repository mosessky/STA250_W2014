Homework 3. Part One: Kaggle StackOverflow Question Open or Closure Estimation
=================================================
In this part of homework 3, I found an existing method proposed by Marco Lui: http://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/forums/t/3083/sharing-my-solution-ranked-10
So, my intial purpose is to test the suggested method and find better alternatives.
I use train.csv as training data and test.csv as test dataset to compare different algorithms' performance.

precise_predictions.txt: this is the prediction results on test dataset by zzgmuntz

en_precise_predictions.txt: this is the prediction results on test dataset by 

Initial Draft
-------------------------------------------------

Now, I test several initial algorithms (LinearSVM, BernoulliNB, MultinomialNB, NearestCentroid, SGDClassifier, SGDClassifier, L1LinearSVC, PassiveAggressiveClassifier, Perceptron).

The performance score is based on F scores (balanced result among accuracy and recall).

In simple words, I found NearestCentroid has best F score on test dataset.
-------------------------------------------------

  
![Figure 1-1](compare_alg.png?raw=true)
Seen from above figure, NearnestCentroid and BernoulliNB has desired score compared with other algorithms.

Homework 3. Part Two: Implementation of Probabilistic Matrix Factorization (PMF) Algorithm to estimate netflix user movie rating
=======================================================
In the second part of homework, according to the homework requirement, I "implement" PMF algorithm and use it to estimate netflix user movie rating.

Original Paper on this topic could be find here: http://www.cs.toronto.edu/~rsalakhu/papers/bpmf.pdf
-----------------------------------------------
