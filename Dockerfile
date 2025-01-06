FROM python:3.10

RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH" \
    POETRY_VERSION=1.8.5

RUN curl -sSL https://install.python-poetry.org | python3 -  --version $POETRY_VERSION

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN poetry install --no-root

COPY . /app

ENTRYPOINT ["poetry", "run"]
