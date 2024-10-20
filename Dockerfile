FROM --platform=linux/amd64 python:3.10-slim AS python-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        build-essential


RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR $PYSETUP_PATH

COPY poetry.lock pyproject.toml ./

RUN poetry install --no-root

FROM python-base as production

COPY --from=python-base $PYSETUP_PATH $PYSETUP_PATH

WORKDIR /app

COPY . .

#CMD ["sleep", "10000"]
ENTRYPOINT ["bash","entrypoint.sh"]
