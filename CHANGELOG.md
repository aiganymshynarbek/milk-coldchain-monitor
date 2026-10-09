# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/) and the
project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-10

### Added
- Point forecasters: persistence, moving average, linear trend and
  ARIMA(5,1,0) with sliding-window ridge estimation and stability correction.
- Excursion early warning: labels, ROC-AUC, average precision, threshold at a
  fixed false-alarm rate, event-level detection metrics.
- Look-ahead-free feature construction and the Tier-2 volatility trigger.
- Evaluation on held-out logs, figures, command-line interface `coldchain`.
- Synthetic cold-store generator, CSV input/output.
- Unit tests, Ruff configuration, GitHub Actions CI and release workflows.
