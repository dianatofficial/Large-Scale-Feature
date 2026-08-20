.PHONY: help install install-dev test test-cov lint format clean run-pipeline run-dashboard docker-up docker-down

help:
	@echo "Available Commands:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development and test dependencies"
	@echo "  make test             Execute test suite"
	@echo "  make test-cov         Execute tests with coverage report"
	@echo "  make lint             Run code linters"
	@echo "  make format           Auto-format code with black and isort"
	@echo "  make run-pipeline     Execute the batch feature transformation pipeline"
	@echo "  make run-dashboard    Launch the Streamlit interactive dashboard"
	@echo "  make docker-up        Spin up full distributed cluster"
	@echo "  make docker-down      Tear down distributed cluster"
	@echo "  make clean            Remove cache and temporary artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

test-cov:
	pytest --cov=src --cov-report=term-missing --cov-report=html tests/

lint:
	flake8 src tests --max-line-length=100 --extend-ignore=E203,W503
	mypy src --ignore-missing-imports

format:
	isort src tests scripts config
	black src tests scripts config

run-pipeline:
	python scripts/run_batch_job.py --config config/local.yaml

run-dashboard:
	streamlit run src/ui/app.py --server.port=8501

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info metastore_db derby.log spark-warehouse
