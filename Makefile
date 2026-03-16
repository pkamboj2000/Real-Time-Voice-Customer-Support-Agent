.PHONY: install run test lint build-index evaluate simulate docker-build docker-run clean

install:
	pip install -r requirements.txt

run:
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ scripts/

build-index:
	python scripts/build_index.py

evaluate:
	python scripts/evaluate.py

simulate:
	python scripts/simulate_call.py

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf data/vector_index
