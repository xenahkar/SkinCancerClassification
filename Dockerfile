FROM python:3.10-slim

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

RUN poetry install

COPY . /app

CMD ["poetry", "run", "python", "commands.py"]
