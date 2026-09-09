.PHONY: help setup test run-app optimize benchmark docker-build docker-run clean

help:
	@echo "Comandos disponibles en LogisticNetwork:"
	@echo "  make setup        - Instala dependencias y prepara el entorno"
	@echo "  make test         - Ejecuta las pruebas unitarias con pytest"
	@echo "  make optimize     - Ejecuta el motor MILP prescriptivo"
	@echo "  make benchmark    - Ejecuta la heurística naive de referencia"
	@echo "  make run-app      - Inicia la aplicación de Streamlit"
	@echo "  make docker-build - Construye la imagen Docker del proyecto"
	@echo "  make docker-run   - Corre el contenedor con el dashboard en el puerto 8501"
	@echo "  make clean        - Limpia archivos temporales de Python y cachés"

setup:
	pip install -e .

test:
	pytest tests/

optimize:
	python -m src.optimization.solver

benchmark:
	python -m src.optimization.benchmark

run-app:
	streamlit run app/app.py

docker-build:
	docker build -t logistic-network:latest .

docker-run:
	docker run -p 8501:8501 logistic-network:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
