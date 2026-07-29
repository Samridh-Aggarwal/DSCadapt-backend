FROM python:3.11-slim

RUN useradd -m -u 1000 user

# System libraries WeasyPrint needs: Pango for text layout, cairo for drawing,
# gdk-pixbuf for images, shared-mime-info so it can identify what it is given.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libcairo2 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY --chown=user . /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["uvicorn", "app:api", "--host", "0.0.0.0", "--port", "7860"]
