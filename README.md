# PROJ_MAYJ_DSA4262 [Public Repository]
DSA4262 Project to Build a Classification Model to Detect M6A Modification in RNA Transcript Reads

## Group Members:
- Amas Lua
- Yan Jing Yi
- Michael Yang
- Chua Yong Sing

## Overview of how we build the model:

- Step 1: **Parse information from data.json** files into a usable format

- Step 2: **Feature extraction and data transformations**
    - Added a Read_Counts column, which corresponds to the number of reads for each transcript at each candidate m6A position.
    - Split the Sequence column into 3 columns: first_base, last_base and middle_sequence.
    - For the column middle_sequence, we converted the categorical variable using the OneHotEncoder function in the sklearn package.
    - For the columns first_base and last_base, we converted the categorical variables by label coding each variable using a mapper

- Step 3: **Train-test split by gene_id** that can be found in data.info to make sure no overlapping of genes between different split.
    - Categorised the genes into 3 categories based on the genes transcripts counts: Low, Medium and High
    - Splitting of dataset
        - Training set: 30% of the genes from each category
        - Test set: remaining 70% of the genes from each category
    - Combine the genes in all 3 categories
    - Split the transcripts reads using gene_id to obtain our test set and training set
    - Resampling of the minority class is done on the training set and test set to deal with the imbalanced dataset

- Step 4: Build a baseline XGBoost Model with resampled data from Step 3

## How to use our model built to run predictions:
1. Provision a Ubuntu 20.04 Large Instance on Research Gateway and SSH into it.

2. **ONLY IF USING POWERSHELL**
   ```bash
   ssh -i '<path/to/pemfile.pem>' -L 8888:localhost:8888 ubuntu@<ip-address of Instance>
   ```

3. CD into a working directory that your Instance is mounted to where you want to run our model inference in.

4. Clone our public repository into your working directory:

   Insert your Github personal access token in the command below. Click [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) to create your token.
    ```bash
    git clone https://<tokenhere>@github.com/jingyiyanlol/PROJ_MAYJ_DSA4262.git
    ```
    
5. CD into the cloned repository:
    ```bash
    cd PROJ_MAYJ_DSA4262
    ```
    
6. If you have not installed Python on your Instance, install it:
    ```bash
    sudo apt update
    sudo apt install python3-pip python3-dev
    ```
    
7. Install virtual environment, create, and activate one
    ```bash
    sudo -H pip3 install --upgrade pip
    sudo -H pip3 install virtualenv
    virtualenv ~/.venv
    source ~/.venv/bin/activate
    ```
    
8. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    
9. Run our model's prediction on the `small_test_data.json` that we have provided for you in this repository:
    ```bash
    python run_predictions.py XGBoost_v2.pkl small_test_data.json small_test_data_predictions.csv
    ```
10. You should see the similar following outputs in your terminal:
    ```bash
    Loading Model...
    Loading Dataset Features...
    Processing Features...
    Predicting Probabilites...
    Directory 'XGBoost_v2_predictions' created!
    time take: 14.066171407699585
    ```
11. The outputs above indicate a successful run of our model predictions. You should see a new directory called `XGBoost_v2_predictions` in your working directory. The directory should contain the following file which contains the output of our model predictions:
    ```bash
    small_test_data_predictions.csv
    ```
