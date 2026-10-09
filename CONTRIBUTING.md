# Contributing

1. Open an issue that describes the bug or the proposed feature.
2. Create a branch from `main`: `git checkout -b feature/short-name`.
3. Install the development tools: `pip install -e ".[dev]"`.
4. Add or update tests together with the code.
5. Before committing run `ruff check .` and `pytest`.
6. Write commit messages in the form `type: summary`
   (`feat`, `fix`, `docs`, `test`, `ci`, `refactor`).
7. Open a pull request; it is merged when the CI pipeline is green.
