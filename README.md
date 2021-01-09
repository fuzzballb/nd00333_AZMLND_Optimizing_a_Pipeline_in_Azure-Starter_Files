# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This is a dataset containing data of a Bank marketing campaign. The label column y contains the result that states if a client is subscribed to the bank product. 

The best performing model was found using AutoML and is MaxAbsScaler LightGBM with an accucracy of 0.9161 This is only slightly higher than the hyperdrive score of 0.9156

## Scikit-learn Pipeline

The first method we used was training with Hyperdrive tuning and letting the Scikit-learn estimator perform the LogisticRegression task. The hyperdrive tuning supplies the estimator with a set of random hyper parameters for which are in a range specified by the RandomParameterSampling. The range for Regulaization penalty (--C) can be any value between 0.2 and 15. Regarding max iterations (--max_iter) the random hyper parameter is either 100 or 300.

ps = RandomParameterSampling( {
        "--C": uniform(0.2, 15),
        "--max_iter": choice( 100, 300)
    }
)

Because the random values are selected from the entire search space, there is a good chance that good hyper parameters could be found. Although ideally the search space around the best found value should be explored more instead of continuing searching the whole space with random values

The BanditPolicy makes sure that runs with low accuracy are terminated early to save costly compute time

## AutoML
AutoML uses different Machine learning algorithms like RandomForest, LightGBM, GradientBoosting etc. Regarding hyper parameters, these are set automatically by AutoML and also the features selection and scaling is done automatically. If there are issues with your data f.e. High cardinality (a lot of unique values) that would make it hard to use for training, then AutoML gives a warning.

## Pipeline comparison
The difference between the algorithm that was found by AutoML ( MaxAbsScaler LightGBM accucracy 0.9161) and the Hyperdrive (Logistic regression accuracy 0.9156) was very minor. This can be because the Hyperdrive got lucky with selecting the right random value and/or that the data and corresponding label was well suited for logistic regression. 

## Future work
As stated under Scikit-learn Pipeline it would help if the RandomParameterSampling of hyperdrive would be extended with continuing to search in the area of the best random samples.

