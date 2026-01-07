@echo off
echo 🔧 Borrando entorno virtual anterior...
rmdir /s /q .venv

echo 🧪 Creando nuevo entorno virtual...
python -m venv .venv

echo ✅ Activando entorno...
call .venv\Scripts\activate.bat

echo 📦 Instalando dependencias...
pip install flask==2.2.3 werkzeug==2.2.3 requests==2.28.2 opencv-python==4.11.0.86 flask-cors openai==0.28.1 python-dotenv faiss-cpu sentence-transformers numpy transformers==4.37.2

echo 🧪 Verificando instalación de tokenizer...
python -c "from transformers import GPT2TokenizerFast; print(GPT2TokenizerFast.from_pretrained('gpt2').tokenize('Verificación completada'))"

echo ✅ Entorno virtual listo. Puedes ahora correr tu servidor.
pause