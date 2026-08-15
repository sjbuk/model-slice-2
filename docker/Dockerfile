FROM python:3.11-slim

WORKDIR /app

# assimp-utils provides the `assimp` CLI, used to convert uploaded FBX/GLB
# models to OBJ -- the pipeline's loader (meshpartition/mesh_io.py) only
# reads OBJ, and pulling in Blender for this would balloon the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends assimp-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["pytest", "-v"]
