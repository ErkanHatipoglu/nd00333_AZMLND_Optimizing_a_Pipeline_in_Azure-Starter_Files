# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset is about a phone call marketing campaign. The original data can be found [@UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). The dataset can be used (we are using) to predict if the client will subscribe to a term deposit or not. The target variable is y. 

Citation: [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

The best performing model in the AutoML part of the project was a VotingEnsemble with an accuracy score of 0.9140. The details will be described below. 

## Scikit-learn Pipeline
**The Architecture**

In the first part of the project we are using a training script (train.py) that gets the data from WEB as a TabularDatasetFactory Class object. The goal is to optimize the [LogisticRegression estimator](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) from [scikit learn library](https://scikit-learn.org/stable/index.html) using Azure Machine Learning [HyperDrive package](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?preserve-view=true&view=azure-ml-py). For this very purpose, the training script also contains a clean_data function that preprocess the data (TabularDatasetFactory object) and returns two pandas data frames as the predictors and the target.

As described above the dataset is about a phone call marketing campaign. It has 20 predictors (features) and a target. The explanation for the dataset predictors and the target can be found [@UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

We are trying the optimize two of the LogisticRegression hyperparameters: 

The first hyperparameter is called 'C' which is a numerical value of float. 'C' is the inverse of regularization strength and must be a positive float. Smaller values specify stronger regularization. The default value for 'C' is 1.0.

The second hyperparameter is called 'max_iter' which is a numerical value of int. 'max_iter' is the maximum number of iterations taken for the solvers to converge. The default value for 'max_iter' is 100.

A typical hyperparameter tuning by using Azure Machine Learning has several [steps](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters). These are:

* Define the parameter search space,
    * We are using [RandomParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py) which defines a  random sampling over a hyperparameter search space.
    * We are using [uniform](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py#uniform-min-value--max-value-) for obtaining 'C' values that will be used in hyperparameter tuning. Uniform specifies a uniform distribution from which samples are taken.
    * We are using [choice](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py#choice--options-) for obtaining 'max_iter' values that will be used in hyperparameter tuning. Choice specifies a discrete set of options to sample from. 
* Specify a primary metric to optimize,
    * We are using 'accuracy' as our primary metric.
* Specify early termination policy for low-performing runs,
    * We are using [BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py) with evaluation_interval=4, and slack_factor=0.08
        * Bandit policy defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation.
        * evaluation_interval is the frequency for applying the policy.
        * slack_factor is the ratio used to calculate the allowed distance from the best performing experiment run.
* Allocate resources,
    * We are allocating a [compute cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python#what-is-a-compute-cluster) with vm_size='STANDARD_D2_V2', and max_nodes=4
* Launch an experiment with the defined configuration
* Visualize the training runs
* Select the best configuration for your model

The HyperDrive is controlled via the udacity-project.ipynb notebook. The flow is described below:

* Initialize the Workspace and the experiment,
* Create a compute cluster with vm_size='STANDARD_D2_V2' and max_nodes=4,
* Specify a parameter sampler,
* Specify a Policy,
* Create a sklearn estimator for use with train.py,
* Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy,
* Submit the hyperdrive run to the experiment,
* Get the best run and save the model from that run.

**The Parameter sampler**

The parameter sampler defines our search space. We have used uniform for 'C' and choice for 'max_iter' as parameter_expressions. parameter_expressions defines *functions* that can be used in HyperDrive to describe a hyperparameter search space. These *functions* are used to specify different types of hyperparameter distributions. uniform is selected for for 'C' because 'C' requires a continuous set of values. choice is selected for 'max_iter' because 'max_iter' requires a discrete set of values.

**The Early Stopping Policy**

An early termination policy specifies that if you have a certain number of failures, HyperDrive will stop looking for the answer.  As a result we terminate poorly performing runs with an early termination policy. Early termination improves computational efficiency. We have used evaluation_interval=4 which means that after the forth interval, the best performing run will be compared with the upcoming runs' scores, and if they are smaller than best performing run - slack_factor(which is 0.08 for our case) the run will be canceled.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
