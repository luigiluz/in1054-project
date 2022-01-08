PYTHON_MODULES := in1054
PYTHONPATH := .
VENV := venv
BIN := $(VENV)/bin

PYTHON := env PYTHONPATH=$(PYTHONPATH) $(BIN)/python
PIP := $(BIN)/pip

REQUIREMENTS := -r requirements.txt
PRE_COMMIT := $(BIN)/pre-commit

bootstrap: venv \
			requirements \

clean:
	rm -r $(VENV)

venv:
	python3 -m venv $(VENV)

requirements:
	$(PIP) install $(REQUIREMENTS)

parse:
	$(PYTHON) scripts/parser_script.py

preprocess:
	$(PYTHON) scripts/preprocessing_script.py

eval_first_stage:
	$(PYTHON) scripts/first_stage_eval_script.py

train_second_stage:
	$(PYTHON) scripts/second_stage_train_script.py

create_text_vectorizer:
	$(PYTHON) scripts/create_text_vectorizer_script.py

test_second_stage:
	$(PYTHON) scripts/second_stage_test_script.py