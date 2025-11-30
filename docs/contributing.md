---
title: Contributing
summary: Get involved with zarrwhals development
description: Guide for contributing to zarrwhals including development setup, testing, and documentation.
keywords: contributing, development, testing, documentation, pull request
order: 5
---

# Contributing

Your help is welcome!

Get started contributing to zarrwhals. We assume familiarity with git and GitHub pull requests.

For more extensive tutorials, see [pyOpenSci](https://www.pyopensci.org/learn.html), [Scientific Python](https://learn.scientific-python.org/development/tutorials/), or the [scanpy developer guide](https://scanpy.readthedocs.io/en/latest/dev/index.html).

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Optionally, [mise](https://mise.jdx.dev/) provides task shortcuts.

/// tab | uv
    new: true

    :::bash
    uv venv -p 3.13
    uv sync --all-groups
///

/// tab | mise

    :::bash
    uv venv -p 3.13
    mise run install
///

This creates a `.venv` with all dev, test, doc, and bench dependencies.

---

## Code Style

We use [prek](https://github.com/j178/prek) for consistent formatting and other checks. Run checks locally:

/// tab | uv
    new: true

    :::bash
    uvx prek run --all-files
///

/// tab | mise

    :::bash
    mise run check
///

Or run linting/formatting separately:

/// tab | uv
    new: true

    :::bash
    # Check for issues
    uvx ruff check src/ tests/

    # Format code
    uvx ruff format src/ tests/
///

/// tab | mise

    :::bash
    mise run lint
    mise run format
///

---

## Testing

We use [pytest](https://docs.pytest.org/) for testing.

/// tab | uv
    new: true

    :::bash
    uv run pytest tests/ -v
///

/// tab | mise

    :::bash
    mise run test
///

With coverage:

/// tab | uv
    new: true

    :::bash
    uv run pytest tests/ --cov=zarrwhals --cov-report=html
///

/// tab | mise

    :::bash
    mise run test:cov
///

Coverage report: `htmlcov/index.html`

### CI

GitHub Actions runs tests on Python 3.11–3.13 for all PRs. A separate job tests against pre-release dependencies. See `.github/workflows/test.yaml`.

---

## Documentation

We use [MkDocs](https://www.mkdocs.org/) with [mkdocs-shadcn](https://asiffer.github.io/mkdocs-shadcn/), [mkdocstrings](https://mkdocstrings.github.io/), and [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).

/// tab | uv
    new: true

    Build:

    :::bash
    uv run mkdocs build

    Serve with auto-reload:

    :::bash
    uv run mkdocs serve
///

/// tab | mise

    Build:

    :::bash
    mise run docs

    Serve with auto-reload:

    :::bash
    mise run docs:serve
///

Local server runs at `http://127.0.0.1:8000`.

---

## Releases

Follow [Semantic Versioning](https://semver.org/). We use [uv-ship](https://github.com/TLouf/uv-ship) and [git-cliff](https://git-cliff.org/) for automation.

/// tab | mise (recommended)
    new: true

    :::bash
    # Patch release (0.0.x)
    mise run release:patch

    # Minor release (0.x.0)
    mise run release:minor

    # Major release (x.0.0)
    mise run release:major
///

/// tab | manual

    1. Update version in `pyproject.toml`
    2. Generate changelog: `mise run changelog`
    3. Commit changes
    4. Create GitHub release with tag `vX.X.X`
///

GitHub releases auto-publish to [PyPI](https://pypi.org/).

## AI Usage

I don't care if you use AI or not, just keep [this comment in mind][ai-mindfullness].

[ai-mindfullness]: https://github.com/ocaml/ocaml/pull/14369#issuecomment-3556593972
