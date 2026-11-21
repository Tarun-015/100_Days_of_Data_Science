    3. What are training and test datasets?
Now let's connect sampling to machine learning.

Suppose you have 10,000 customer records and want to predict whether a customer will default on a loan.

You divide the dataset into two parts:
Original dataset
10,000
Customer records




Training set

8,000
80% of data

Test set

2,000
20% of data


Training
The model learns patterns from these 8,000 records.

Testing
The model is evaluated on 2,000 records it did not train on.


Why do we do this?

Because a model might memorize the training data instead of learning patterns that generalize to new observations.

The test dataset helps us estimate how well the model performs on unseen data.
4. The most important distinction: Training vs Validation vs Testing
In practical machine learning, we often use three datasets.
Dataset	Purpose
Training	Learn model parameters
Validation	Tune hyperparameters and choose models
Testing	Final unbiased evaluation

For example, imagine you're developing a Random Forest model.
1. Train the model using training data.
2. Change n_estimators, max_depth, etc., using validation performance.
3. Select the best configuration.
4. Evaluate the final model on test data.
Important: If you repeatedly check test performance and make decisions based on it, you are indirectly using the test data for model selection. Your test set is no longer a truly untouched evaluation set.
