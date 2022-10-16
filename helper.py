##################################################
# Import Dependencies                            #
##################################################
import sys
import pandas as pd
import random
import numpy as np
import json

# Preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Models
import xgboost as xgb 
from xgboost import XGBClassifier

# plot feature importance
from xgboost import plot_importance

# Sklearn Model Metrics
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score)

##################################################
# Helper Functions                               #
##################################################
def convert_json_to_dataframe(data_path):
    """
    convert_json_to_dataframe function takes in a path of a JSON file and returns a Pandas DataFrame with columns:
    "transcript_id", "transcript_position", "transcript_reads"
    """
    # Read in JSON file
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    # Create DataFrame
    df = pd.DataFrame({"Transcript": data})
    df['transcript_id'] = df['Transcript'].apply(lambda x: list(x.keys())[0])
    df['transcript_position'] = df['Transcript'].apply(lambda x: list(x[list(x.keys())[0]].keys())[0])
    df['transcript_reads'] = df['Transcript'].apply(lambda x: x[list(x.keys())[0]][list(x[list(x.keys())[0]].keys())[0]])
    df.drop(columns=['Transcript'], inplace=True)
    
    return df

def convert_json_to_dataframe_v2(data_path):
    """
    convert_json_to_dataframe_v2 function:
    - takes in a path of a JSON file and returns a Pandas DataFrame with columns:
        ["Transcript", "Position" "Sequence", "Read Counts", "dwelling_time(-1)", "std_dev(-1)", "mean_current(-1)", 
        "dwelling_time(0)", "std_dev(0)", "mean_current(0)", "dwelling_time(+1)", "std_dev(+1)", "mean_current(+1)"]
    """
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame({"Transcript": data})
    
    def extract_transcript_reads(row):
        transcript = row['Transcript']
        transcript_name = list(transcript.keys())[0]
        transcript_dict = {}
        transcript_dict["Transcript"] = transcript_name

        reads = transcript[transcript_name]

        # get the transcript position
        transcript_position = list(reads.keys())[0]
        transcript_dict["Position"] = int(transcript_position)

        # get rna sequence of that position
        rna_seq = list(reads[transcript_position].keys())[0] 
        transcript_dict["Sequence"] = rna_seq

        # get the number of reads
        num_reads = len(reads[transcript_position][rna_seq])
        transcript_dict["Read_Counts"] = num_reads

        # put nested list of extract features into reads_dict
        transcript_dict["Features"] = reads[transcript_position][rna_seq]
        return transcript_dict

    df1 = df.assign(Transcript=df.apply(extract_transcript_reads, axis=1))
    df1 = df1["Transcript"].apply(pd.Series)
    df2 = df1.explode("Features")
    features = ["dwelling_time(-1)", "std_dev(-1)", "mean_current(-1)", "dwelling_time(0)", "std_dev(0)", "mean_current(0)", "dwelling_time(+1)", "std_dev(+1)", "mean_current(+1)"]

    df2 = df2.reset_index(drop=True)
    
    # new df from the column of lists
    split_df = pd.DataFrame(df2['Features'].tolist(), columns=features)

    # combine split_df to df2
    df3 = pd.concat([df2, split_df], axis=1)
    df3 = df3.drop(columns=['Features'])
    
    return df3


def convert_data_frame_to_json(data_frame, json_output_path):
    """
    convert_data_frame_to_json takes in a dataframe and output path of the JSON file
    and converts each row in the dataframe into a dictionary object and writes each dictionary
    as a line in the JSON output file.
    """
    json_list = []
    for index, row in data_frame.iterrows():
        transcript_id, transcript_position, transcript_reads = row['transcript_id'], row['transcript_position'], row['transcript_reads']
        transcript_dict = {transcript_id: {transcript_position: transcript_reads}}
        json_list.append(transcript_dict)
        
    with open(json_output_path, 'w') as fp:
        fp.write(
            '\n'.join(json.dumps(i) for i in json_list) + '\n')
        
def train_test_split(data_info, data, train_size = 0.7, random_state = 4262):
    """
    train_test_split takes in data_info dataframe and dataframe of the output format of convert_json_to_dataframe.
    It splits the data into train and test by gene_id and returns the train and test dataframes and corresponding labels dataframes.
    """
    
    data['transcript_position']=data['transcript_position'].astype(int)
    data_with_gene_id = pd.merge(data, data_info, on=['transcript_id', 'transcript_position'], how='left')
    
    random.seed(random_state)
    
    def get_counts_class(row):
        count = row['transcripts_count']
        if count <= 14:
            return 'Low'
        if ((count >14) and (count <= 45)):
            return 'Medium'
        else:
            return 'High'
        
    gene_transcripts_counts = data_info.groupby('gene_id').count().reset_index()[["gene_id", "transcript_id"]].rename({"transcript_id": "transcripts_count"}, axis=1)    
    gene_transcripts_counts['counts_class'] = gene_transcripts_counts.apply(get_counts_class, axis=1)
    
    # Sample 30% of genes from each count class to be in test set and 70% in train set
    high_class = gene_transcripts_counts[gene_transcripts_counts['counts_class'] == 'High']
    medium_class = gene_transcripts_counts[gene_transcripts_counts['counts_class'] == 'Medium']
    low_class = gene_transcripts_counts[gene_transcripts_counts['counts_class'] == 'Low']
    train_high = random.sample(high_class['gene_id'].tolist(), int(len(high_class)*train_size))
    test_high = [x for x in high_class['gene_id'].tolist() if x not in train_high]
    train_medium = random.sample(medium_class['gene_id'].tolist(), int(len(medium_class)*train_size))
    test_medium = [x for x in medium_class['gene_id'].tolist() if x not in train_medium]
    train_low = random.sample(low_class['gene_id'].tolist(), int(len(low_class)*train_size))
    test_low = [x for x in low_class['gene_id'].tolist() if x not in train_low]

    train_genes = train_high + train_medium + train_low
    test_genes = test_high + test_medium + test_low
    
    train_data = data_with_gene_id[data_with_gene_id['gene_id'].isin(train_genes)][["transcript_id", "transcript_position", "transcript_reads"]]
    test_data = data_with_gene_id[data_with_gene_id['gene_id'].isin(test_genes)][["transcript_id", "transcript_position", "transcript_reads"]]

    train_data_labels = data_info[data_info['gene_id'].isin(train_genes)]
    test_data_labels = data_info[data_info['gene_id'].isin(test_genes)]
    
    return train_data, test_data, train_data_labels, test_data_labels   


def pre_process_data_v1(data_unlabelled, data_info):
    """
    pre_process_data_v1 function:
    - takes in data_unlabelled dataframe and data_info dataframe
    - extracts out middle sequence (5 bases) of original sequence and add labels from data.info to unlablelled data
    - extracts out first base and last base of original sequence and maps the bases A, T, G, C to 0, 1, 2, 3
    - returns a dataframe with columns:
        ['gene_id', 'transcript_id', 'transcript_position', 'first_base', 'last_base', 'middle_sequence', 'Read_Counts', 
        'dwelling_time(-1)', 'std_dev(-1)', 'mean_current(-1)',
        'dwelling_time(0)', 'std_dev(0)', 'mean_current(0)',
        'dwelling_time(+1)', 'std_dev(+1)', 'mean_current(+1)', 'label']
    """
    ATGC_mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    
    data_unlabelled['middle_sequence'] = data_unlabelled.apply(lambda row: row["Sequence"][1:6], axis = 1)
    data_unlabelled['first_base'] = data_unlabelled.apply(lambda row: ATGC_mapping[row["Sequence"][0]], axis = 1)
    data_unlabelled['last_base'] = data_unlabelled.apply(lambda row: ATGC_mapping[row["Sequence"][6]], axis = 1)
    
    # merge data_unlabelled and data_info
    data_with_labels = pd.merge(data_unlabelled, data_info, left_on=['Transcript', 'Position'], right_on=['transcript_id', 'transcript_position'], how='left')
    data_with_labels.drop(['Transcript', 'Position'], axis=1, inplace=True)

    columns_names = ['gene_id', 'transcript_id', 'transcript_position', 'first_base', 'last_base', 'middle_sequence', 'Read_Counts', 'dwelling_time(-1)', 'std_dev(-1)', 'mean_current(-1)','dwelling_time(0)', 'std_dev(0)', 'mean_current(0)','dwelling_time(+1)', 'std_dev(+1)', 'mean_current(+1)', 'label']
    data_with_labels = data_with_labels[columns_names]

    return data_with_labels

def pre_process_data_v2(data):
    """
    pre_process_data_v2 function:
    - takes in a dataframe that is the output of the function convert_json_to_dataframe_v2
    - extracts out middle sequence (5 bases) of original sequence and add labels from data.info to unlablelled data
    - extracts out first base and last base of original sequence and maps the bases A, T, G, C to 0, 1, 2, 3
    - returns a dataframe with columns:
        ['transcript_id', 'transcript_position', 'first_base', 'last_base', 'middle_sequence', 'Read_Counts', 
        'dwelling_time(-1)', 'std_dev(-1)', 'mean_current(-1)',
        'dwelling_time(0)', 'std_dev(0)', 'mean_current(0)',
        'dwelling_time(+1)', 'std_dev(+1)', 'mean_current(+1)']
    """
    ATGC_mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    
    data['middle_sequence'] = data.apply(lambda row: row["Sequence"][1:6], axis = 1)
    data['first_base'] = data.apply(lambda row: ATGC_mapping[row["Sequence"][0]], axis = 1)
    data['last_base'] = data.apply(lambda row: ATGC_mapping[row["Sequence"][6]], axis = 1)
    
    # rename columns 'Transcript' and 'Position'
    data.rename(columns={'Transcript': 'transcript_id', 'Position': 'transcript_position'}, inplace=True)
    # print("data columns: {}".format(data.info()))
    
    columns_names = ['transcript_id', 'transcript_position', 'first_base', 'last_base', 'middle_sequence', 'Read_Counts', 'dwelling_time(-1)', 'std_dev(-1)', 'mean_current(-1)','dwelling_time(0)', 'std_dev(0)', 'mean_current(0)','dwelling_time(+1)', 'std_dev(+1)', 'mean_current(+1)']
    data_processed = data[columns_names]

    return data_processed

def one_hot_encode_train_data(train_data):
    """
    one_hot_encode_train_data function:
    - takes in data frame from pre_process_data_v1 function and returns a one hot encoded x_train matrix, y_train matrix and transformer object
    - ['gene_id', 'transcript_id', 'transcript_position', 'label'] are dropped from the input dataframe
    - the feature 'middle_sequence' is one-hot encoded
    """
    x_train = train_data.drop(['gene_id', 'transcript_id', 'transcript_position', 'label'], axis=1)
    y_train = train_data['label']
    
    # convert categorical features to str type (not sure why XGboost cannot handle categorical type)
    categorical_features = ['middle_sequence']
    for col in categorical_features:
        x_train[col] = x_train[col].astype('category')
    
    # Fit column transformer on train x
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", dtype='int')
    ct = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, ["middle_sequence"])
        ],
        remainder='passthrough'
    )
    
    ct.fit(x_train) # fit column transformer on train x
    X_train = ct.transform(x_train) # One-hot encoded X_train matrix
    
    return X_train, y_train, ct
    
def one_hot_encode_test_data(test_data, transformer):
    """
    one_hot_encode_test_data function:
    - takes in data frame from pre_process_data_v1 function and transformer object from one_hot_encode_train_data function 
    and returns a one hot encoded x_test matrix and y_test matrix
    """
    x_test = test_data.drop(['gene_id', 'transcript_id', 'transcript_position', 'label'], axis=1)
    y_test = test_data['label']
    
    # convert categorical features to str type (not sure why XGboost cannot handle categorical type)
    categorical_features = ['middle_sequence']
    for col in categorical_features:
        x_test[col] = x_test[col].astype('category')
    
    X_test = transformer.transform(x_test) # One-hot encoded X_test matrix
    
    return X_test, y_test

def one_hot_encode_test_data_v2(test_data, transformer):
    """
    one_hot_encode_test_data function:
    - takes in data frame from pre_process_data_v2 function and transformer object from one_hot_encode_train_data function 
        and returns a one hot encoded x_test matrix
    """
    x_test = test_data.drop(['transcript_id', 'transcript_position'], axis=1)
    
    # convert categorical features to str type (not sure why XGboost cannot handle categorical type)
    categorical_features = ['middle_sequence']
    for col in categorical_features:
        x_test[col] = x_test[col].astype('category')
    
    X_test = transformer.transform(x_test) # One-hot encoded X_test matrix
    
    return X_test

def plot_feature_importance(column_transformer, model, model_name, output_dir):
    """
    plot_feature_importance function:
    - takes in column transformer object, model object, model name and output directory of the plot
    - plots the feature importance of the model and saves it in the output_dir
    """
    X_train_columns = column_transformer.get_feature_names_out().tolist()
    fig_name = output_dir + "/" + model_name + "_feature_importance.jpg"
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X_train_columns)), columns=['Value','Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('Features importance -' + model_name)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()
    
def get_predict_probability(test_data, X_test, model):
    """
    get_predict_probability function:
    - takes in test_data from pre_process_data_v1 function, X_test matrix from one_hot_encode_test_data function and model object
    - returns a dataframe with columns ['gene_id', 'transcript_id', 'transcript_position', 'label', 'predicted_prob']
    """
    
    test_data_with_predicted_prob = test_data.copy()
    y_pred_prob = model.predict_proba(X_test)
    test_data_with_predicted_prob['predicted_prob'] = y_pred_prob[:,1]
    
    # Getting the mean probability score for each gene 
    test_data_with_predicted_prob_mean = test_data_with_predicted_prob.groupby(['gene_id', 'transcript_id', 'transcript_position', 'label'])['predicted_prob'].mean().reset_index()
    
    return test_data_with_predicted_prob_mean

def get_predict_probability_v2(test_data, X_test, model):
    """
    get_predict_probability function:
    - takes in test_data from pre_process_data_v2 function, X_test matrix from one_hot_encode_test_data_v2 function and model object
    - returns a dataframe with columns ['transcript_id', 'transcript_position', 'score']
    """
    test_data_with_predicted_prob = test_data.copy()
    y_pred_prob = model.predict_proba(X_test)
    test_data_with_predicted_prob['score'] = y_pred_prob[:,1]
    
    # Getting the mean probability score for each gene 
    test_data_with_predicted_prob_mean = test_data_with_predicted_prob.groupby(['transcript_id', 'transcript_position'])['score'].mean().reset_index()
    
    return test_data_with_predicted_prob_mean

def get_predicted_label(data_with_prob, threshold):
    """
    get_predicted_label function:
    - takes in data_with_prob from get_predict_probability function and a threshold value
    - reteurns a dataframe with columns ['gene_id', 'transcript_id', 'transcript_position', 'label', 'predicted_prob','predicted_label']
    """
    data_with_pred_labels = data_with_prob.copy()
    data_with_pred_labels['predicted_label'] = data_with_pred_labels['predicted_prob'].apply(lambda x: 1 if x > threshold else 0)
    return data_with_pred_labels

def get_acccuracy_metrics(y_true, y_score, y_pred, model_name):
    """
    get_acccuracy_metrics function:
    - takes in y_true, y_score, y_pred and model_name
    - returns a dataframe with columns ['model', 'precision','recall', 'F1_score', 'accuracy', 'roc_auc', 'pr_auc', 'false_positive', 'false_negative', 'true_positive', 'true_negative']
    """
    accuracy_metric_df = pd.DataFrame(columns=['model', 'precision','recall', 'F1_score', 'accuracy', 'roc_auc', 'pr_auc', 'false_positive', 'false_negative', 'true_positive', 'true_negative'])
    
    cf_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    F1_score = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_score)
    false_positive = cf_matrix[1][0]
    false_negative = cf_matrix[0][1]
    true_positive = cf_matrix[1][1]
    true_negative = cf_matrix[0][0]

    accuracy_metric_df.loc[0] = [model_name, precision, recall, F1_score, accuracy, roc_auc, pr_auc, false_negative, false_positive, true_positive, true_negative]
    return accuracy_metric_df