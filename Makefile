.PHONY: help setup test run-app clean

help:
	@echo "Comandos disponibles:"
	@echo "  make setup     - Instala dependencias y prepara el entorno"
	@echo "  make test      - Ejecuta las pruebas unitarias con pytest"
	@echo "  make run-app   - Inicia la aplicación de Streamlit"
	@echo "  make clean     - Limpia archivos temporales de Python y cachés"

setup:
	pip install -e .

test:
	pytest tests/

run-app:
	streamlit run app/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
