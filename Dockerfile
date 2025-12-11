FROM python:3.11-slim

# Install ffmpeg for audio playback
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot source
COPY bot.py ./

# Run as non-root user
RUN useradd -m botuser
USER botuser

# Environment variable placeholder (override at runtime)
ENV DISCORD_TOKEN=""

CMD ["python", "bot.py"]
