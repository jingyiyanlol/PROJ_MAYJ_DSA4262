# PROJECT MAYJ DSA4262 PROJECT REPOSITORY
This project is aimed at building a *Classification Model* to detect post-transcriptional M6A modification in RNA Transcript reads
![image](https://user-images.githubusercontent.com/92244042/199272325-0c6ada6a-f554-47e3-9111-7d2f53ef727b.png)
Image source: [Spandidos Publicatios](https://www.spandidos-publications.com/10.3892/ijmm.2020.4746)

### Group Members: 
**M**ichael Yang, **A**mas Lua, **Y**ong Sing Chua, **J**ing Yi Yan

## Overview of how we build our model:

- **Step 1: Parse information from data.json** files into pandas dataframe for further processing

- **Step 2: Feature extraction and data transformations**
    - Added a Read_Counts column, which corresponds to the number of reads for each transcript at each candidate m6A position.
    - Split the Sequence column into 3 columns: first_base, last_base and middle_sequence.
    - For the column middle_sequence, we converted the categorical variable using the OneHotEncoder function in the sklearn package.
    - For the columns first_base and last_base, we converted the categorical variables by label coding each variable using a mapper

- **Step 3: Train-test split by gene_id** that can be found in data.info to make sure no overlapping of genes between different split.
    - Categorised the genes into 3 categories based on the genes transcripts counts: Low, Medium and High
    - Splitting of dataset
        - Training set: 30% of the genes from each category
        - Test set: remaining 70% of the genes from each category
    - Combine the genes in all 3 categories
    - Split the transcripts reads using gene_id to obtain our test set and training set
    - Resampling of the minority class is done on the training set and test set to deal with the imbalanced dataset

- **Step 4: Build a baseline XGBoost Model** with resampled data from Step 3

- **Step 5: Experimenting with different values of hyper-parameters** such as `max_depth`, `learning_rate`, `n_estimators`, `reg_alpha`, `reg_lambda` and tracking the differences with *MLFlow GUI*. More information about how we used MLFlow can be found in the model_training folder.

- **Step 6: Choose best model.** The model that has the most improvements in the `auc_roc` and `pr_auc` metrics from our baseline model is chosen as our final model.
    | Model       | pr_auc     | roc_auc |
    | :---        |   :----:   | :----:  |
    | XGBoost_v1  | 0.837      | 0.31    |
    | XGBoost_v2  | 0.847      | 0.309   |


## How to use our model to get predictions:
* **Note**: `XGBoost_v1.pkl` in this directory was the model that we built for our first submission, while `XGBoost_v2.pkl` was the model that we built for our second submission. The model that we built for our second submission is the model that you will be using to test our inference pipeline.

### 1. Provision a *Ubuntu 20.04 Large* Instance on *Research Gateway* and **SSH** into it to use the Linux terminal. We recommend an EBS Volume Size of **1500** and an instance Type of **4xlarge** for faster results.

- Your IP address of your instance can be found by following the steps in the screenshot below:

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

    - Replace `123.456.789.542` with your IP instance's IP address as well as the `~\path\to\pem-file.pem` and paste the following configurations into your `config` file:
        ```config
        Host 123.456.789.542
            HostName 123.456.789.542
            User ubuntu
            IdentityFile ~\path\to\pem-file.pem
            Port 22
        ```

    - Now, Click the `F1` Key or `Fn + F1` Keys again to connect to the SSH Host you have just configured.

        ![image](https://user-images.githubusercontent.com/92244042/199288300-b34d2a98-851f-4380-9a81-b0be30f5e509.png)

    - A New VSCode Windows will be launched and you can select `Linux` when prompted to choose between `Linux`, `Windows`, or `Mac`.

### 3. CD into a working directory that your Instance is mounted to where you want to clone our repository to and run our model inference in.
- For example in terminal:
    ```bash
    cd <working_dir>
    ```
- In VSCode:

    ![image](https://user-images.githubusercontent.com/92244042/199289851-afa237a9-30db-4484-8f7c-63d807d5e34d.png)


### 4. Clone our public repository into your working directory:

- Using Git in terminal:
```bash
git clone https://github.com/jingyiyanlol/PROJ_MAYJ_DSA4262.git
```

- If the above method does not work, you can lick [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) to learn how to create your GitHub personal access token and try the next method.

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
- If u are prompted with the message `Do you want to continue? [Y/n]`, type `Y` and press enter.

    - If you would like to install our dependencies in a *python virtual environment*, run the commands below instead:
    ```bash
    make install
    ```
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

### 9. You should see the similar following outputs in your terminal if the run is succesful:
```bash
Predicting labels of small test dataset...
Loading Model...
Loading Dataset Features...
Processing Features...
Predicting Probabilities...
Time taken: 00:00:17
Prediction of small test dataset complete!
```
### 10. If the run is successful, you should see a new directory called `XGBoost_v2_predictions` created in your working directory. The directory should contain the file `small_test_data_predictions.csv` which contains the output of our model predictions

**INSERT SCREENSHOT HERE ONCE MODEL V2 READY**