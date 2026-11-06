"""Print the API's OpenAPI schema as JSON. The frontend generates its types from it.

    uv run python scripts/export_openapi.py > frontend/openapi.json
"""
import json
import os
import tempfile

from autotune.presentation.api.app import create_app


def main() -> None:
    # create_app opens the job database, so point it at a throwaway directory.
    with tempfile.TemporaryDirectory() as data_dir:
        os.environ["AUTOTUNE_DATA_DIR"] = data_dir
        schema = create_app().openapi()
    print(json.dumps(schema, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
