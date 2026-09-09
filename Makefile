format:
	uv run isort .
	uv run black .
	uv run pylint --recursive=y tests deployment model-building monitoring

unit_test:
	uv run pytest tests/unit_tests

integration_test:
	bash ./tests/integration_tests/run.sh

run: format unit_test integration_test
	echo HO GYA BHAI
