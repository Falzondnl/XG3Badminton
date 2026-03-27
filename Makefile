.PHONY: help install test test-cov lint fmt check lock clean run docker docker-test

PYTHON ?= python
PORT   ?= 8009

help:
	@echo "XG3 Badminton Platform — make targets"
	@echo ""
	@echo "  install       Install all dependencies"
	@echo "  test          Run test suite"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  lint          Run ruff linter"
	@echo "  fmt           Auto-format with black"
	@echo "  check         lint + test"
	@echo "  lock          Update regression lock state"
	@echo "  verify        Verify regression lock passes"
	@echo "  clean         Remove __pycache__, .pyc, coverage artefacts"
	@echo "  run           Start dev server (port $(PORT))"
	@echo "  docker        Build Docker image"
	@echo "  docker-test   Run tests inside Docker"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -x --tb=short -q

test-cov:
	$(PYTHON) -m pytest tests/ \
		--cov=. \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-fail-under=80 \
		-x --tb=short -q

lint:
	ruff check .

fmt:
	black .

check: lint test

lock:
	$(PYTHON) scripts/lock_regression_state.py

verify:
	$(PYTHON) scripts/lock_regression_state.py --verify

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov coverage.xml .pytest_cache .mypy_cache .ruff_cache

run:
	uvicorn main:app --host 0.0.0.0 --port $(PORT) --reload

docker:
	docker build -t xg3-badminton:latest .

docker-test:
	docker-compose --profile test run --rm test
