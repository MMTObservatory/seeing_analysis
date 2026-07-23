PYTHON ?= /Users/tim/conda/envs/mmtwfs/bin/python

.PHONY: help
help:
	@echo "Usage:"
	@echo "  make <period>   generate figures, e.g. make 2026q2  or  make 2025"
	@echo "  make test       run the unit tests"

.PHONY: test
test:
	$(PYTHON) -m pytest tests/

# Period specs (YYYY or YYYYqN) all start with "20".
# No file named after the target is ever created, so this re-runs every time.
20%:
	$(PYTHON) -m seeing_summary $@
