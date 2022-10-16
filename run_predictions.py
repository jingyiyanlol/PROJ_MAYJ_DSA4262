##################################################
# IMPORT DEPENDENCIES
##################################################
# Data processing
import os
import sys
import pandas as pd
import joblib
import helper as h

# Model
import xgboost as xgb 
from xgboost import XGBClassifier

import time


if __name__ == '__main__':
    s = time.time()
    ##################################################
    # Load Model
    ##################################################
    model_pkl_path = sys.argv[1]
    # modeL_pkl_path = "../model_training/XGBoost_version1.pkl"
    
    print("Loading Model...")
    model_pkl = joblib.load(model_pkl_path)
    model, transformer = model_pkl
    
    ##################################################
    # Load Features (dataset1.json/ dataset2.json)
    ##################################################
    test_data_path = sys.argv[2]
    # test_data_path = ../data/small_test_data.json
    # test_data_path = ../data/dataset1.json
    # test_data_path = ../data/dataset2.json
    
    print("Loading Dataset Features...")
    test_data_raw = h.convert_json_to_dataframe_v2(test_data_path)
    
    ##################################################
    # Processsing features
    ##################################################
    print("Processing Features...")
    test_data = h.pre_process_data_v2(test_data_raw)
    X_test = h.one_hot_encode_test_data_v2(test_data, transformer)
    
    ##################################################
    # Generating predictions
    ##################################################
    print("Predicting Probabilities...")
    predictions_df = h.get_predict_probability_v2(test_data, X_test, model)
    
    ##################################################
    # Output predictions to CSV
    ##################################################
    prediction_output_file_name = sys.argv[3]
    # prediction_output_file_name = "small_test_data.csv"
    # prediction_output_file_name = "mayj_dataset1_1.csv"
    # prediction_output_file_name = "mayj_dataset2_1.csv"
    
    predictions_folder_name = model_pkl_path.split("/")[-1][:-4] + '_predictions' 
    isExist = os.path.exists(predictions_folder_name)
    if not isExist:
        os.mkdir(predictions_folder_name)
        print("Directory '{}' created!".format(predictions_folder_name))
        
    prediction_output_file_path = os.path.join(predictions_folder_name, prediction_output_file_name)
    predictions_df.to_csv(prediction_output_file_path, index=False)
    
    e = time.time()
    print("time taken: {}".format(e-s))
    
# SAMPLE COMMAND TO RUN THE SCRIPT:
"""
py run_predictions.py ../model_training/XGBoost_version1.pkl ../data/small_test_data.json small_test_data.csv
py run_predictions.py ../model_training/XGBoost_version1.pkl ../data/dataset1.json mayj_dataset1_1.csv
py run_predictions.py ../model_training/XGBoost_version1.pkl ../data/dataset2.json mayj_dataset2_1.csv
"""
    
    