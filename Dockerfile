FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY teldiag ./teldiag
COPY tests ./tests
RUN pip install --no-cache-dir ".[dev]"

# Headless by default: score the diagnoser and fail the container if it regresses.
ENTRYPOINT ["teldiag"]
CMD ["evaluate", "--runs-per-class", "20", "--min-accuracy", "0.95", "--max-fp-rate", "0.0", "--out", "reports/metrics.json"]
