FROM python:3.12-slim

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código
COPY . .

# Exponer el puerto (Railway lo asigna dinámicamente)
EXPOSE 8000

# Ejecutar con Gunicorn para Flask
CMD gunicorn --bind 0.0.0.0:$PORT app:app