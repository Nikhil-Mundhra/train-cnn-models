SHELL := /bin/bash

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(abspath $(VENV)/bin/python)
PIP := $(VENV_PYTHON) -m pip
PYTEST := $(VENV_PYTHON) -m pytest

ARCH ?= convnextv2_base
TASK ?= multi_head
IMG_SIZE ?= 0

.PHONY: help venv install test smoke-test train clean

help:
	@echo "Train CNN Models Suite"
	@echo ""
	@echo "Targets:"
	@echo "  make venv            Create virtual environment"
	@echo "  make install         Install Python training dependencies"
	@echo "  make test            Run diagnostic and unit test suite"
	@echo "  make smoke-test      Run quick smoke test on training pipeline"
	@echo "  make train           Run full training (use ARCH=... TASK=... IMG_SIZE=...)"
	@echo "  make clean           Remove cache and temporary build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make train ARCH=resnet50"
	@echo "  make train TASK=segmentation ARCH=hierarchical_unet IMG_SIZE=512"
	@echo "  make smoke-test ARCH=convnext_small"
	@echo ""
	@echo "Environment variables:"
	@echo "  OCT_LOCAL_DEVICE=cpu|mps|cuda   Override compute device selection (default: auto)"

venv:
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		$(PYTHON) -m venv "$(VENV)"; \
		$(PIP) install --upgrade pip; \
	fi

install: venv
	@$(PIP) install -r image-classification-model-training/requirements.txt

test:
	@$(PYTHON) -m pytest tests/diagnostics

smoke-test:
	@echo "Running smoke test on unified training pipeline (ARCH=$(ARCH), TASK=$(TASK))..."
	KMP_DUPLICATE_LIB_OK=TRUE $(PYTHON) train.py \
		--task $(TASK) --arch $(ARCH) --img-size $(IMG_SIZE) \
		--smoke-test --epochs-warmup 1 --epochs-finetune 1 --batch-size 8

train:
	@echo "Launching training pipeline (ARCH=$(ARCH), TASK=$(TASK))..."
	KMP_DUPLICATE_LIB_OK=TRUE $(PYTHON) train.py \
		--task $(TASK) --arch $(ARCH) --img-size $(IMG_SIZE)

train-convnext: train

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .pytest_cache .coverage
