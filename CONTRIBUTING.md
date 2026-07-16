# Contributing to RetinaAI

Thank you for considering contributing to RetinaAI! Please read this guide to help us maintain a high-quality research codebase.

## How to Contribute

1. **Fork the repository** on GitHub.
2. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the coding standards below.
4. **Test your changes** locally:
   ```bash
   # Run tests
   pytest tests/
   # Run training/evaluation scripts to ensure nothing breaks
   python scripts/train_unetpp.py --help
   ```
5. **Commit your changes** with a clear, descriptive message.
6. **Push to your fork** and open a Pull Request (PR) against the `research/curriculum-topo-loss` branch.

## Reporting Issues

Please use the GitHub Issues tracker to report bugs or request features. Include:
- A clear, descriptive title.
- Steps to reproduce the issue.
- Expected vs. actual behavior.
- Relevant logs, error messages, or screenshots.
- Your environment (OS, Python version, GPU, etc.).

## Pull Request Process

1. Ensure your PR passes all existing tests.
2. Update the README or documentation if your changes affect usage.
3. Keep PRs focused: one feature or bug fix per PR.
4. Write a clear PR description explaining the problem and solution.
5. Respond to reviewer feedback promptly.
6. After approval, a maintainer will merge your PR.

## Coding Standards

- Follow [PEP 8](https://pep8.org/) for Python code.
- Use descriptive variable and function names.
- Add docstrings to all public functions and classes.
- Keep functions focused and under 50 lines when possible.
- Comment complex logic, but avoid obvious comments.
- Run `flake8` and `black` locally before submitting:
  ```bash
  flake8 src/ scripts/
  black src/ scripts/
  ```
- Ensure new code includes appropriate unit tests in the `tests/` directory.

## Research Specific Guidelines

- When adding new experiments, update `docs/RESEARCH_PLAN.md` and the ablation table if applicable.
- New metrics or loss functions should be documented in `docs/README_UNETPP.md` and the relevant source files (e.g., `models/losses_unetpp.py`).
- Update requirements files (`requirements_unetpp.txt`) if you add new dependencies.
- Keep the `README.md` up-to-date with any changes to installation, usage, or expected results.

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

## Getting Help

If you need help, feel free to:
- Ask questions in the Issues tab.
- Contact the maintainer directly via GitHub.
- Consult the existing documentation in the `docs/` directory.

Thank you again for contributing to RetinaAI!