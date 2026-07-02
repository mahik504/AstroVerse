.PHONY: install test lint demo train-small evaluate dashboard clean

# Install all dependencies
install:
	cd research/evonex && pip install -r requirements.txt
	cd apps/astrolens-web && npm install

# Quickstart Demo (End-to-End)
demo:
	@echo "Running AstroVerse Quickstart Demo..."
	cd research/evonex && python src/build_dataset.py --version demo --catalog ../datasets/toi_catalog.csv
	cd research/evonex && python src/run_baselines.py

# Train on a small config (sanity check)
train-small:
	@echo "Running training sweep on small dataset..."
	cd research/evonex && python src/train_evomoe.py --config configs/small.yaml

# Run full baseline & ablation evaluation
evaluate:
	@echo "Evaluating EvoMoE against baselines..."
	cd research/evonex && python src/run_baselines.py
	@echo "Running ablation studies..."
	cd research/evonex && python src/run_ablations.py

# Launch AstroLens Web Dashboard
dashboard:
	@echo "Starting API and Next.js UI..."
	cd services/evonex-api && start /B python -m uvicorn main:app --port 8000
	cd apps/astrolens-web && npm run dev

# Run all tests
test:
	cd research/evonex && python -m pytest tests/ -v
	cd services/evonex-api && python -m pytest tests/ -v

# Clean generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
