.PHONY: install test lint smoke dev-api dev-web clean

# Install all dependencies
install:
	cd research/evonex && pip install -r requirements.txt
	cd apps/astrolens-web && npm install

# Run all tests
test:
	cd research/evonex && python -m pytest tests/ -v
	cd services/evonex-api && python -m pytest tests/ -v

# Run model smoke test
smoke:
	cd research/evonex && python src/model_evonex.py

# Lint Python code
lint:
	ruff check research/evonex/src/ services/evonex-api/

# Start API dev server
dev-api:
	cd services/evonex-api && python -m uvicorn main:app --reload --port 8000

# Start frontend dev server
dev-web:
	cd apps/astrolens-web && npm run dev

# Train with default config
train:
	cd research/evonex && python src/train_evomoe.py --config configs/default.yaml

# Train with small config (quick test)
train-small:
	cd research/evonex && python src/train_evomoe.py --config configs/small.yaml

# Evaluate model
evaluate:
	cd research/evonex && python src/evaluate_evomoe.py

# Clean generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
