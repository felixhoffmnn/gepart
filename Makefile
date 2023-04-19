.PHONY: help black mypy
.DEFAULT_GOAL := help

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  black           to run black"
	@echo "  mypy            to run mypy"

black:
	@echo "Running black..."
	poetry run black .

mypy:
	@echo "Running mypy..."
	poetry run mypy ./studienarbeit
