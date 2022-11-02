# Model Training and Evaluations Logging with XGBoost, SciKit-Learn, and MLflow

## 1. Train Test Split
The script `train_test_split.py` splits the data into training and test sets. The outputs `train_data.json`, `train_data.info`, `test_data.json`, `test_data.info` are saved in the `data` folder.

**Note**: Before this script is ran, a `data` directory containing `data.json` and `data.info` was created in the same level as the script. 

## 2. XGBoost Baseline Model Training, Evaluation, and Logging with MLflow
The script `XGBoost.py` trains a baseline XGBoost model, evaluates the model, and logs the model and evaluation metrics with MLflow. 

The model artefact `XGBoost_no_tuning.pkl` is saved in a `models` folder created by the script. This pkl file is then renamed and uploaded as `XGBoost_v1.pkl` that you see in our main directory of this repostory.

The feature importance plot `xgboost_no_tuning_feature_importance.jpg` is also automatically saved in a `feature_importance_plots` folder created by the script.

## 3. XGBoost Hyperparameter Tuning and Logging with MLflow

As Cross Validation is very heavy on computation, we realised that it would be better to select the best hyperparameters by fitting our model and evaluating them straight away. This way, we saved time and resources.

- **Step 1**: 

    Using the `XGBoost_random_params.py` script that takes in a value for `max_depth`, `learning_rate`, `n_estimators`, `reg_alpha`, and `reg_lambda` to narrow our grid search. 
    
    The script fits the XGBoost Classifier with the given parameters and logs the model's evaluation on our test dataset to MLflow. 
    
    This approach allows us to quickly decide whether to increase or decrease the value of a hyperparameter based on the evaluation metric since training time of a single model is faster than training time of multiple models.

- **Step 2**: 

    Using `hyper_parameters_grid_search.py` script which has a grid of hyperparameters defined in the script. 
    
    The script iterates through the grid and prints out the best parameters and the corresponding `auc_roc` score. 
    
    With the parameters returned from `hyper_parameters_grid_search.py`, we input them into another script `XGBoost_tuning.py` which fits the XGBoost Classifier with the best hyperparameters and logs the model's evaluation on our test dataset to MLflow. The model artefact and features importance plot are similarly saved automatically as described in the previous [section]() and uploaded as `XGBoost_v2.pkl` in this repository.

The following screenshot displays the *MLflow GUI* of the model experiments that we have ran: 

![image](https://user-images.githubusercontent.com/92244042/199523417-6c80cb65-955e-40ed-b8f0-79e85a01f529.png)

The 2 rows that were highlighted respectively represents our first baseline model and our final tuned model. This GUI can be fired up with the command `mlflow ui` in the terminal and will be running on `port 5000` by default unless specified otherwise.