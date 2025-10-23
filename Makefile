.PHONY: help build-ext test flowchart clean

help:
	@echo "Available targets:"
	@echo "  build-ext       Build Cython extensions in place"
	@echo "  test            Run pytest suite"
	@echo "  flowchart       Regenerate flowchart PNG from LaTeX source"
	@echo "  clean           Remove build artifacts"

build-ext:
	python setup.py build_ext --inplace

test:
	uv run pytest

flowchart:
	@cd docs && bash regenerate_flowchart.sh

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	cd docs && rm -f flowchart.pdf flowchart.aux flowchart.log
