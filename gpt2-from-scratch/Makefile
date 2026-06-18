.PHONY: install test data train generate

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

data:
	python scripts/download_data.py

train:
	python -m gpt2_scratch.train --config configs/tiny_shakespeare.json

generate:
	python -m gpt2_scratch.generate --prompt "ROMEO:"
