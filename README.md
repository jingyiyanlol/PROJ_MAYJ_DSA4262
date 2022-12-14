[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
# NUS DSA4262 AY2022/2023 Semester 1 PROJECT REPOSITORY
## Team MAYJ
A group of final year students majoring in Data Science and Analytics at the National University of Singapore (NUS)

### Group Members: 
[**M**ichael Yang](https://github.com/yangzichang12), [**A**mas Lua](https://github.com/amaslua), [**Y**ong Sing Chua](https://github.com/yongsingc), [**J**ing Yi Yan](https://github.com/jingyiyanlol)

## Project Motivation
This project is aimed at building a *Classification Model* to detect post-transcriptional m6A modification in RNA Transcript reads. The probability of m6A modification of a given candidate RNA-Seq short read coming from a specific transcript position is being output as an inference of the model. 

m6A modification (m6A RNA methylation) is the most abundant modification among the many post-transcriptional modifications discovered so far as it serves as a significant regulator of transcript expression. 

During the modification, adenosine molecules are being methylated, thereby resulting in a change in its chemical structure that translates to possibly destructive effects on one's metabolism. When the rate of m6A modification is too high, oncoprotein expression may be increased, thereby triggering cancer cells proliferation and mestastasis. 

Hence, it is crucial for us to investigate into means to detect such modifications to allow early detection of illnesses and open up the possibilities of immunotherapy in cancer treatments.

If you would like access to our full training data (`data.json` and `data.info`), please contact our Professor [Jonathan Goeke](https://github.com/jonathangoeke) for access and approval.

![image](https://user-images.githubusercontent.com/92244042/199272325-0c6ada6a-f554-47e3-9111-7d2f53ef727b.png)
Image source: [Spandidos Publicatios](https://www.spandidos-publications.com/10.3892/ijmm.2020.4746)


### Project Video:
[![Youtube Video Thumbnail](https://user-images.githubusercontent.com/92244042/200112460-6fe7aef3-bfa9-4202-aa5c-b922128ad504.png)](https://youtu.be/LqMW0TfXFJ8)

## Overview of how we built our model:

![image](https://user-images.githubusercontent.com/92244042/199651248-9871b55e-e464-40e8-9c34-8e22042116a1.png)

- **Step 1: Parse information from provided data.json** file into pandas DataFrame for further processing

    **Before**: 
    ```json
    {"ENST00000000233":
        {"244":
            {"AAGACCA":[[0.00299,2.06,125.0,0.0177,10.4,122.0,0.0093,10.9,84.1],
                        [0.00631,2.53,125.0,0.00844,4.67,126.0,0.0103,6.3,80.9],
                        [0.00465,3.92,109.0,0.0136,12.0,124.0,0.00498,2.13,79.6]]
            }
        }
    }
    ```
    
    **After**:

    **Note**: We added a `Read_Counts` column, which corresponds to the number of reads for each transcript at each candidate m6A position.

    |   | Transcript 	  | Position | Sequence	| Read_Counts |	dwelling_time(-1) |	std_dev(-1)	| mean_current(-1) | dwelling_time(0) |	std_dev(0) | mean_current(0) | dwelling_time(+1) | std_dev(+1)	| mean_current(+1) |
    | - | :---            | :---:    | :---:    | :---:       | :---:             | :---:       | :---:            | :---:            | :---:      | :---:           | :---:             | :---:        | :---:            |
    | 0 | ENST00000000233 | 244	     | AAGACCA  | 185	      | 0.00299           | 2.06        | 125.0            | 0.01770          | 10.40      | 122.0           | 0.00930           | 10.90        | 84.1             |
    | 1 | ENST00000000233 | 244	     | AAGACCA  | 185	      | 0.00631           | 2.53        | 125.0            | 0.00844          | 4.67       | 126.0           | 0.01030           | 6.30	        | 80.9             |
    | 2	| ENST00000000233 | 244	     | AAGACCA  | 185	      | 0.00465           | 3.92        | 109.0            | 0.01360          | 12.00      | 124.0           | 0.00498           | 2.13	        | 79.6             |

- **Step 2: Train-test split by gene_id** that can be found in data.info to make sure no overlapping of genes between different split, preventing data leakage
    - Categorised the genes into 3 categories based on the genes transcripts counts: *Low*, *Medium* and *High*
      - Low: < 15
      - Medium: 14 - 45
      - High: > 45
    - Distribution of split:
        - Training set: 70% of the genes from each category
        - Test set: remaining 30% of the genes from each category
    - Concatenate the gene_id samples of all 3 categories into 2 lists respectively: *train_genes* and *test_genes*

- **Step 3: Feature extraction and data transformations**
    - Split the `Sequence` column into 3 columns: `first_base`, `last_base` and `middle_sequence`. For the columns `first_base` and `last_base`, we converted the alphabetical letters into numeric numbers with a mapping dictionary.
        ```python
        ATGC_mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        ```
    - Join labels (response variable) from `data.info` file into the dataframe.

    - `data.info` sample rows:
        ```csv
        gene_id,transcript_id,transcript_position,label
        ENSG00000004059,ENST00000000233,244,0
        ENSG00000004059,ENST00000000233,261,0
        ENSG00000004059,ENST00000000233,316,0
        ```

    - Combined DataFrame:

        |   | gene_id         | transcript_id 	| transcript_position | first_base | last_base | middle_sequence | Read_Counts | dwelling_time(-1) | std_dev(-1) | mean_current(-1) | dwelling_time(0) | std_dev(0) | mean_current(0) | dwelling_time(+1) | std_dev(+1) | mean_current(+1) | label |
        | - | :---            | :---            | :---:               |  :---:     |  :---:    | :---:           | :---:       | :---:             | :---:       | :---:            | :---:            | :---:      | :---:           | :---:             | :---:       | :---:            | :---: |
        | 0 | ENSG00000004059 | ENST00000000233 | 244	              | 0          | 0         | AGACC           | 185	       | 0.00299           | 2.06        | 125.0            | 0.01770          | 10.40      | 122.0           | 0.00930           | 10.90       | 84.1             | 0     |
        | 1 | ENSG00000004059 | ENST00000000233 | 244	              | 0          | 0         | AGACC           | 185	       |  0.00631          | 2.53        | 125.0            | 0.00844          | 4.67       | 126.0           | 0.01030           | 6.30	    | 80.9             | 0     |
        | 2	| ENSG00000004059 | ENST00000000233 | 244	              | 0          | 0         | AGACC           | 185	       | 0.00465           | 3.92        | 109.0            | 0.01360          | 12.00      | 124.0           | 0.00498           | 2.13	    | 79.6             | 0     |
    
    - With the gene_id lists from *step 2*, we split the dataframe into training set and test set.
    - For the training set dataframe, we resampled the rows with label of the minority class to deal with the imbalanced dataset (m6A modified transcripts are much less than the unmodified transcripts).
    - After resampling, we dropped identity columns which are namely `gene_id` and `transcript_id`. The `Read_Counts` column was also dropped as it is not logical to use it as a feature to predict m6A modification. 
    - For the column `middle_sequence`, we converted the categorical variable using the `OneHotEncoder` function in the `sklearn` package before fitting the model as XGBoost **cannot** be fitted with values of character type.
    

- **Step 4: Build a baseline XGBoost Model** with resampled training data from *Step 3*

- **Step 5: Experimenting with different values of hyper-parameters** such as `max_depth`, `learning_rate`, `n_estimators`, `reg_alpha`, `reg_lambda` and tracking the differences with *MLflow GUI*. More information about how we used MLflow can be found in the [model_training folder](https://github.com/jingyiyanlol/PROJ_MAYJ_DSA4262/tree/main/model_training).

- **Step 6: Choose best model.** Among the parameters that we tried, we chose the combination that yielded the most improvements in the `auc_roc` and `pr_auc` metrics from our first model as our final model.
    | Model       | pr_auc | roc_auc  | precision | recall | f1_score |
    | :---        | :----: | :----:   | :----:    | :----: | :----:   |
    | XGBoost_v1  | 0.31   | 0.837    | 0.128     | 0.768  | 0.22     |
    | XGBoost_v2  | 0.31   | **0.847**| 0.124     | 0.789  | 0.214    |


## How to use our model to get predictions:
* **Note**: `XGBoost_v1.pkl` in this directory was the model that we built for our first submission, while `XGBoost_v2.pkl` was the model that we built for our second submission. The model that we built for our second submission is the model that you will be using to test our inference pipeline.

![image](https://user-images.githubusercontent.com/92244042/199652315-71751eda-a5ac-4cd5-a6cf-c9cef524bd78.png)

### 1. Provision a *Ubuntu 20.04 Large* Instance on *Research Gateway* and **SSH** into it to use the Linux terminal. 

We recommend an EBS Volume Size of **1500GB** and an Instance Type of **4xlarge** for faster results. This size is also able to handle the workload of predicting the possibilites of m6A modification of the short reads in the [SG-NEx Samples](https://sg-nex-data.s3.amazonaws.com/index.html#data/processed_data/m6Anet/) in reasonable time.

- The IP address of your instance can be found by following the steps in the screenshot below:

    ![image](https://user-images.githubusercontent.com/92244042/199281130-d52a2884-1a52-48a2-9c9a-d6cf9a0ae18f.png)

- **Alternative SSH Method 1: Using local Windows PowerShell**

    You can use the following command format to SSH into the instance using Windows PowerShell:
    ```bash
    ssh -i '<path/to/pemfile.pem>' -L 8888:localhost:8888 ubuntu@<ip-address of Instance>
    ```

    ![image](https://user-images.githubusercontent.com/92244042/199283291-b9183e09-c877-440d-a63d-ad49ac984392.png)

- **Alternative SSH Method 2: Using Visual Studio Code's Remote-SSH extension**

    - Download Remote SSH Extension if you do not have it. You can refer to the screenshot below to check if you have the extension installed.

        ![image](https://user-images.githubusercontent.com/92244042/199285330-b4807f34-0cc3-41fa-b7f0-97ce7f1bf5b1.png)

    - Click the `F1` Key or `Fn + F1` Keys to launch the search bar for you to configure your SSH Host details.

        ![image](https://user-images.githubusercontent.com/92244042/199287365-70d8121e-9956-4ff4-ab0a-00004d582a2e.png)

    - Replace `123.456.789.542` with your instance's IP address as well as the `~\path\to\pem-file.pem` and paste the following configurations into your `config` file:
        ```config
        Host 123.456.789.542
            HostName 123.456.789.542
            User ubuntu
            IdentityFile ~\path\to\pem-file.pem
            Port 22
        ```

    - Now, Click the `F1` Key or `Fn + F1` Keys again to connect to the SSH Host you have just configured.

        ![image](https://user-images.githubusercontent.com/92244042/199288300-b34d2a98-851f-4380-9a81-b0be30f5e509.png)

    - A new VSCode window will be launched and you can select `Linux` when prompted to choose between `Linux`, `Windows`, or `Mac`.

### 3. CD into a working directory that your Instance is mounted to, where you want to clone our repository to and run our model inference in.
- For example in terminal:
    ```bash
    cd <working_dir>
    ```
- In VSCode:

    ![image](https://user-images.githubusercontent.com/92244042/199289851-afa237a9-30db-4484-8f7c-63d807d5e34d.png)


### 4. Clone our public repository into your working directory:

- Using Git in terminal:
```bash
git clone --depth 1 https://github.com/jingyiyanlol/PROJ_MAYJ_DSA4262.git
```

- If the above method does not work, you can try configuring your Git with commands below before running the git clone command above again:
   ```bash
   set GIT_TRACE_PACKET=1
   set GIT_TRACE=1
   set GIT_CURL_VERBOSE=1
   git config --global core.compression 0
   ```

- If the above methods still do not work, you can click [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) to learn how to create your GitHub personal access token and try the next method. If you are unable to clone this repository still, feel free to raise an issue in this repository!

    - Insert your GitHub personal access token in `<tokenhere>` in the command below. 
        ```bash
        git clone https://<tokenhere>@github.com/jingyiyanlol/PROJ_MAYJ_DSA4262.git
        ```
    
### 5. CD into the cloned repository:
- Via terminal:
    ```bash
    cd PROJ_MAYJ_DSA4262
    ```
- Launch Integrated Terminal in VSCode:

    ![image](https://user-images.githubusercontent.com/92244042/199292798-c7f2b206-abdd-4b35-a130-aedc2d008e40.png)

### 6. Update your Ubuntu and install Make using the commands below in order to run our makefile:
```bash
sudo apt update
sudo apt install make
```

### 7. Run the command below to install the required packages to run our model:
```bash
make install_all
```
- If you are prompted with the message `Do you want to continue? [Y/n]`, type `Y` and press `Enter`.

- If you would like to install our dependencies in a *python virtual environment*, run the sequence of commands below instead:
    ```bash
    make install
    ```
    - Make sure that the Python version installed in your machine is 3.8.10 by typing the command:
        ```bash
        python3 --version
        ```
    - Create and activate the virtual environment `venv` by running the commands below:
        ```bash
        python3 -m venv ~/.venv
        source ~/.venv/bin/activate
        ```
    ```bash
    make install_python_dependencies
    ```
    
### 8. Run our model's prediction on the `small_test_data.json` that we have provided for you in this repository:
```bash
make predictions_on_small_dataset
```

### 9. You should see the similar following outputs in your terminal if the run is successful. 

When type the command `ls`, you should see a new directory called `XGBoost_v2_predictions` created in your working directory. The directory should contain the file `small_test_data_predictions.csv` which contains the output of our model predictions with the following columns: `transcript_id`, `transcript_position` and `score`.

![image](https://user-images.githubusercontent.com/92244042/199395555-76a7c646-5b27-4af6-abf2-8de18011be99.png)

### 10. Use our model to do prediction on a dataset that you have. 

You can type the following command in your terminal and replace `<path/to/data.json>` and `<data_predictions.csv>` with the **path to the m6ANet processed RNA sequence reads json file** and the **name of the output csv** file respectively.

```bash
python run_predictions.py XGBoost_v2.pkl <path/to/data.json> <data_predictions.csv>
```

You should be able to see the output csv in the `XGBoost_v2_predictions` directory when the run is completed.
