.PHONY: install test train smoke

install:
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest -q

train:
	python train.py

smoke:
	python train.py --max_steps 50 --eval_interval 25 --eval_iters 5 --sample_tokens 120
