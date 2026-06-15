FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN mkdir -p src/genomics_ml && touch src/genomics_ml/__init__.py
# Create a stub for packeging 
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]" 
# This went to dev in pyproject and installed dev aswell as all dependecies 
COPY . .
# Copy the contecnt itself, by that changes to the code dont make 
# rebuilding tiem consuimng as the rebuild starts from here
EXPOSE 8000
CMD ["uvicorn", "genomics_ml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
