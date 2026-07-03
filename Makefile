.PHONY: install seed api frontend test lint compose

install:
	pip install -e ".[dev]"
	pre-commit install
	cd frontend && npm install

# Regenerate the checked-in demo dataset.
seed:
	python scripts/build_seed.py

# Run the API locally (in-memory graph + embedded Chroma, no servers needed).
api:
	uvicorn precedent.api.main:app --reload --port 8080

# Run the developer visualizer (proxies /api to the API above).
frontend:
	cd frontend && npm run dev

test:
	pytest

lint:
	ruff check .
	mypy src/

# Full stack in containers: Neo4j + Chroma + API + frontend.
compose:
	docker compose -f infra/docker-compose.yml up --build
