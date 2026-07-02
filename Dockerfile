FROM python:3.11-slim

WORKDIR /app

COPY research/evonex/src/ /app/research/evonex/src/
COPY research/evonex/requirements.txt /app/research/evonex/requirements.txt
COPY services/evonex-api/ /app/services/evonex-api/

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    torch --index-url https://download.pytorch.org/whl/cpu \
    pandas \
    python-multipart \
    pyyaml

WORKDIR /app/services/evonex-api

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
