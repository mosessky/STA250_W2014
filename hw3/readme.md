Kaggle StackOverflow Homework
=================================================
In this homework, I found an existing method proposed by Marco Lui: http://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/forums/t/3083/sharing-my-solution-ranked-10
So, my intial purpose is to test the suggested method and find better alternatives.
I use train.csv as training data and test.csv to compare different algorithms.
precise_predictions.txt: this is the prediction results on test dataset by zzgmuntz
en_precise_predictions.txt: this is the prediction results on test dataset by 

Initial Draft
-------------------------------------------------

Now, I test an round of initial algorithms (LinearSVM, BernoulliNB, MultinomialNB, NearestCentroid, SGDClassifier, SGDClassifier, L1LinearSVC, PassiveAggressiveClassifier, Perceptron).

The performance score is based on F scores (balanced result among accuracy and recall).

In simple words, I found NearestCentroid has best F score on test dataset.
-------------------------------------------------

Here is the comparison result:
![Figure 1-1](compare_alg.png?raw=true)



