install_linux_commands:
	@sudo apt update
	@echo "Installing Linux commands like wget..."
	@sudo apt install wget
	
install_python:
	@echo "Installing Python..."
	@sudo apt update
	@sudo apt install python3-pip python3-dev
	@echo "Installing Python Virtual Environment..."
	@sudo apt install python3.8-venv

install_python_dependencies:
	@echo "Installing Python Requirements..."
	pip install --upgrade pip --no-warn-script-location &&\
	pip install -r requirements.txt --no-warn-script-location

install: install_linux_commands install_python

install_all: install_linux_commands install_python install_python_dependencies

predictions_on_small_dataset:
	@echo "Predicting labels of small test dataset..."
	@python3 run_predictions.py XGBoost_v2.pkl small_test_data.json small_test_data.csv
	@echo "Prediction of small test dataset complete!"
	@echo "-------------------------------------------------------------------------------------------------------------------------"