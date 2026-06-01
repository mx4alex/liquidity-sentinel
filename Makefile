.PHONY: install run dashboard test lint

install:
	pip install -e ".[dev]"

run:
	sentinel run

dashboard:
	streamlit run dashboard/app.py

test:
	pytest -q

lint:
	ruff check src tests dashboard
