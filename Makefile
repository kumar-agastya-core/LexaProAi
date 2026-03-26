# ─────────────────────────────────────────────────────────────────────────────
# LexaProAi — Makefile
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

PYTHON = .venv/bin/python

.PHONY: help install setup sync-cold sync-hot batch test lint clean reset-db

# ── Default target ────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  LexaProAi — Lexware Invoice Automation"
	@echo ""
	@echo "  First time:"
	@echo "    make install        Install Python dependencies"
	@echo "    make setup          Initialise DB + sync contacts/categories from Lexware"
	@echo ""
	@echo "  Daily use:"
	@echo "    make batch          OCR pre-process + process all PDFs in pdfs/inbox/"
	@echo "    make sync-hot       Pull latest contacts/categories from Lexware (delta)"
	@echo ""
	@echo "  Dev:"
	@echo "    make test PDF=path/to/invoice.pdf   Dry-run a single PDF (nothing posted)"
	@echo "    make test-pdf PDF=path/to/invoice.pdf   Same but PDF-binary mode"
	@echo "    make lint           Syntax-check all Python files"
	@echo "    make clean          Remove __pycache__ and .pyc files"
	@echo "    make reset-db       Delete and recreate the local database"
	@echo "    make sync-cold      Force a full re-sync from Lexware"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

setup:
	$(PYTHON) -c "from agent.db import init_db; init_db(); print('DB initialised.')"
	$(PYTHON) -c "from agent.sync import cold_sync; cold_sync(); print('Cold sync complete.')"

sync-cold:
	$(PYTHON) -c "from agent.sync import cold_sync; cold_sync(); print('Cold sync complete.')"

sync-hot:
	$(PYTHON) -c "from agent.sync import hot_sync; hot_sync(); print('Hot sync complete.')"

# ── Processing ────────────────────────────────────────────────────────────────

batch:
	$(PYTHON) main.py

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	@if [ -z "$(PDF)" ]; then echo "Usage: make test PDF=path/to/invoice.pdf"; exit 1; fi
	$(PYTHON) test.py --text "$(PDF)"

test-pdf:
	@if [ -z "$(PDF)" ]; then echo "Usage: make test-pdf PDF=path/to/invoice.pdf"; exit 1; fi
	$(PYTHON) test.py "$(PDF)"

# ── Dev utilities ─────────────────────────────────────────────────────────────

lint:
	@echo "Checking syntax of all Python files..."
	@find . -name "*.py" \
		-not -path "./.venv/*" \
		-not -path "./__pycache__/*" \
		-not -path "./*/__pycache__/*" | \
		while read f; do \
			$(PYTHON) -m py_compile "$$f" && echo "  OK  $$f" || echo "  ERR $$f"; \
		done
	@echo "Done."

clean:
	@find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
	@echo "Cleaned."

reset-db:
	@read -p "Delete db/lexware.db and rebuild schema? [y/N] " yn; \
		if [ "$$yn" = "y" ] || [ "$$yn" = "Y" ]; then \
			rm -f db/lexware.db db/lexware.db-wal db/lexware.db-shm; \
			$(PYTHON) -c "from agent.db import init_db; init_db()"; \
			echo "Database reset."; \
		fi
