#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = digit-recognizer
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 digit_recognizer
	isort --check --diff --profile black digit_recognizer
	black --check --config pyproject.toml digit_recognizer

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml digit_recognizer

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) digit_recognizer/dataset.py

## Train the model
.PHONY: train
train: requirements
	$(PYTHON_INTERPRETER) digit_recognizer/modeling/train.py

## Make predictions using the trained model
.PHONY: predict
predict: requirements
	$(PYTHON_INTERPRETER) digit_recognizer/modeling/predict.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$(PRINT_HELP_PYSCRIPT)" < $(MAKEFILE_LIST)