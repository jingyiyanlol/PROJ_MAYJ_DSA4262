# PROJ_MAYJ_DSA4262 [Public Repository]
DSA4262 Project to Build a Classification Model to Detect M6A Modification in RNA Transcript Reads

## Group Members:
- Amas Lua
- Yan Jing Yi
- Michael Yang
- Chua Yong Sing

## Overview of how we build the model:

** TO ADD ON TO THIS PART **


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
