.PHONY: install data pairs train smoke

install:
	pip install -e . --no-deps

# download, split, save. CPU only, run once
data:
	python -m episteme.data

# ask, push back, sample replies, build chosen/rejected
pairs:                  
	python -m episteme.pairs

# ORPO on a 50/50 mix
train:
	python -m episteme.train --name run_50_50 --hold-firm-percent 50

smoke:
	python scripts/01_smoke_test.py

# before report
eval-base:
	python -m episteme.evaluate --name base

#after report
eval-tuned:     
	python -m episteme.evaluate --name run_50_50 --adapter checkpoints/run_50_50
