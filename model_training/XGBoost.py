"""
This script is used to train and log the baseline XGBoost model.
"""
##################################################
# Importing in necessary libraries
##################################################
import os
import warnings

import pandas as pd
import joblib

import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

import helper as h

# Model
import xgboost as xgb 

# Sklearn Model Metrics
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score)

##################################################
# PATHS SETUP
##################################################
# Defining the path of the data
TRAIN_DATA_PATH = 'data/train_data.json'
TRAIN_DATA_INFO_PATH = 'data/train_data.info'
TEST_DATA_PATH = 'data/test_data.json'
TEST_DATA_INFO_PATH = 'data/test_data.info'

###################################################
# MAIN DRIVER
###################################################
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # Loading train and test data
    print("Reading in train and test DATA JSON...")
    train_data_raw = h.convert_json_to_dataframe_v2(TRAIN_DATA_PATH)
    test_data_raw = h.convert_json_to_dataframe_v2(TEST_DATA_PATH )
    print("Reading in train and test DATA INFO...")
    train_data_info = pd.read_csv(TRAIN_DATA_INFO_PATH)
    test_data_info = pd.read_csv(TEST_DATA_INFO_PATH)
        
    # Process features
    print("Processing features...")
    train_data = h.pre_process_data_v1(train_data_raw, train_data_info)
    test_data = h.pre_process_data_v1(test_data_raw, test_data_info)

    X_train, y_train, ct = h.upsample_and_one_hot_encode_train_data(train_data)
    X_test, y_test = h.one_hot_encode_test_data(test_data, ct)

    num_features = len(ct.get_feature_names_out())
        
        
    ##################################################
    # MODEL TRAINING AND LOGGING
    ##################################################
    booster = 'gbtree'
    seed = 4262
    
    with mlflow.start_run():
        # Instantiating model with model parameters
        model = xgb.XGBClassifier(booster=booster, random_state=seed)
        
        # Fitting training data to the model
        print("Fitting model...")
        model.fit(X_train, y_train)
        # logging model parameters
        model_parameters = model.get_xgb_params()
        for parameter in model_parameters:
            value = model_parameters[parameter]
            mlflow.log_param(parameter, value)
        mlflow.log_param('num_features', num_features)
        mlflow.log_param('num_train_samples', len(X_train))
        
        # Running prediction on validation dataset
        print("Running prediction...")
        test_data_with_predicted_prob_mean = h.get_predict_probability(test_data, X_test, model)
        test_data_with_pred_labels = h.get_predicted_label(test_data_with_predicted_prob_mean, threshold=0.5)
        y_true, y_score, y_pred = test_data_with_pred_labels['label'], test_data_with_pred_labels['predicted_prob'], test_data_with_pred_labels['predicted_label']
        
        # Getting metrics on the validation dataset
        print("Calculating model metrics...")
        cf_matrix = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        print("precision score: {}".format(precision))
        mlflow.log_metric('precision', precision)
         
        recall = recall_score(y_true, y_pred)
        print("recall: {}".format(recall))
        mlflow.log_metric('recall', recall)
        
        F1_score = f1_score(y_true, y_pred)
        print("F1 score: {}".format(F1_score))
        mlflow.log_metric('F1_score', F1_score)
        
        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy: {}".format(accuracy))
        mlflow.log_metric('accuracy', accuracy)
        
        roc_auc = roc_auc_score(y_true, y_score)
        print("ROC AUC score: {}".format(roc_auc))
        mlflow.log_metric('roc_auc', roc_auc)
        
        pr_auc = average_precision_score(y_true, y_score)
        print("PR AUC score: {}".format(pr_auc))
        mlflow.log_metric('pr_auc', pr_auc)
        
        false_positive = cf_matrix[1][0]
        print("False Positives: {}".format(false_positive))
        mlflow.log_metric('false_positive', false_positive)
        
        false_negative = cf_matrix[0][1]
        print("False Negatives: {}".format(false_negative))
        mlflow.log_metric('false_negative', false_negative)
        
        true_positive = cf_matrix[1][1]
        print("True Positives: {}".format(true_positive))
        mlflow.log_metric('true_positive', true_positive)
        
        true_negative = cf_matrix[0][0]
        print("True Negatives: {}".format(true_negative))
        mlflow.log_metric('true_negative', true_negative)
        
        # Saving feature importance plot directly to S3
        model_name = 'xgboost_no_tuning'
        feature_importance_output_dir = 'feature_importance_plots'

        isExist = os.path.exists(feature_importance_output_dir)
        if not isExist:
            os.mkdir(feature_importance_output_dir)
            print("Directory '{}' created!".format(feature_importance_output_dir))
            
        h.plot_feature_importance(ct, model, model_name, feature_importance_output_dir)
        print("Feature Importance Plot for {} saved!".format(model_name))
        
        # Saving pkl file of the model directly to S3
        models_pkl_output_dir = 'models'
        model_file_name = "XGBoost_no_tuning.pkl"
        model_file_output_path = os.path.join(models_pkl_output_dir, model_file_name)
        isExist = os.path.exists(models_pkl_output_dir)
        if not isExist:
            os.mkdir(models_pkl_output_dir)
            print("Directory '{}' created!".format(models_pkl_output_dir))
            
        joblib.dump((model, ct), model_file_output_path, compress=3)
        print("Model '{}' saved to {}!".format(model_file_name, model_file_output_path))
        
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(sk_model = model,
                             artifact_path = 'XGBoost_no_tuning',
                             registered_model_name = model_name)
        else:
            mlflow.sklearn.log_model(sk_model=model,
                                     artifact_path='XGBoost_no_tuning')
            