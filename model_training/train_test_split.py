##################################################
# Import Dependencies                            #
##################################################
import pandas as pd
import random
import json

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

if __name__ == '__main__':
    
    ##################################################
    # Paths of Inputs and Outputs                    #
    ##################################################
    full_data_path = 'data/data.json'
    data_info_path = 'data/data.info'
    output_dir = 'data'
    train_data_output_path = output_dir + "/train_data.json"
    test_data_output_path = output_dir + "/test_data.json"
    train_data_info_output_path = output_dir + "/train_data.info"
    test_data_info_output_path = output_dir + "/test_data.info"
    
    ##################################################
    # Reading in data and train test split           #
    ##################################################
    data_info = pd.read_csv(data_info_path)
    data = convert_json_to_dataframe(full_data_path)
    print("SPITTING DATA INTO TRAIN AND TEST ...")
    train_data, test_data, train_data_labels, test_data_labels = train_test_split(data_info, data)
    
    ##################################################
    # Saving train and test data                     #
    ##################################################
    print("SAVING TRAIN AND TEST DATA ...")
    convert_data_frame_to_json(train_data, train_data_output_path)
    convert_data_frame_to_json(test_data, test_data_output_path)
    train_data_labels.to_csv(train_data_info_output_path, index=False)
    test_data_labels.to_csv(test_data_info_output_path, index=False)
    print("TRAIN AND TEST DATA SPLITTING AND SAVING COMPLETETED!")
    