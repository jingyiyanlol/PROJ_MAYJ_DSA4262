# PROJ_MAYJ_DSA4262 [Public Repository]
DSA4262 Project to Build a Classification Model to Detect M6A Modification in RNA Transcript Reads

## Group Members:
- Michael Yang
- Amas Lua
- Yong Sing Chua
- Jing Yi Yan

## Overview of how we build the model:

** TO ADD ON TO THIS PART **


## How to use our model built to run predictions:
1. Provision a Ubuntu 20.04 Large instance on Research Gateway and SSH into it.

    **ADD SOME SCREENSHOTS HERE OR SOMETHING**

2. CD into a working directory that your instance is mounted to where you want to run our model inference in.

3. Clone our public repository into your working directory:
    ```bash
    git clone https://github.com/jingyiyanlol/PROJ_MAYJ_DSA4262
    ```
4. CD into the cloned repository:
    ```bash
    cd PROJ_MAYJ_DSA4262
    ```
5. If you have not installed python on your instance, install it:
    ```bash
    sudo apt apt-get update
    sudo apt install python3-pip python3-dev
    ```
6. Install virtual environment, create, and activate one
    ```bash
    sudo -H pip3 install --upgrade pip
    sudo -H pip3 install virtualenv
    virtualenv ~/.venv
    source ~/.venv/bin/activate
    ```
7. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
8. Run our model's prediction on the `small_test_data.json` that we have provided for you in this repository:
    ```bash
    python run_predictions.py XGboost_v2.pkl small_test_data.json small_test_data_predictions.csv
    ```
9. You should see the similar following outputs in your terminal:
    ```bash
    Loading Model...
    Loading Dataset Features...
    Processing Features...
    Predicting Probabilites...
    Directory 'XGBoost_v2_predictions' created!
    time take: 14.066171407699585
    ```
10. The outputs above indicate a successful run of our model predictions. You should see a new directory called `XGBoost_v2_predictions` in your working directory. The directory should contain the following file:
    ```bash
    small_test_data_predictions.csv
    ```