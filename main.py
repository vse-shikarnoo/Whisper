from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
import os
import tempfile
import json
import uuid
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ollama Audio Processing API", version="5.0.0")

# Пробуем загрузить улучшенный процессор с Ollama
try:
    from ollama_audio_processor import OllamaAudioProcessor
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    processor = OllamaAudioProcessor(
        hf_token=HF_TOKEN,
        ollama_model=OLLAMA_MODEL,
        ollama_host=OLLAMA_HOST
    )
    
    logger.info("OllamaAudioProcessor успешно инициализирован")
    
except Exception as e:
    logger.error(f"Ошибка загрузки OllamaAudioProcessor: {e}")
    processor = None

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/process")
async def process_audio_file(
    file: UploadFile = File(...),
    use_ollama: bool = Query(True, description="Использовать Ollama для суммаризации"),
    ollama_model: str = Query(None, description="Модель Ollama (если не указана, используется настройка по умолчанию)")
):
    """
    Обработка аудио/видео файла с поддержкой Ollama
    """
    if not processor:
        raise HTTPException(status_code=500, detail="Audio processor not available")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Проверяем расширение файла
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.mp4', '.avi', '.mov', '.mkv'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="File type not supported")
    
    # Временно меняем модель Ollama если указана
    original_model = processor.ollama_model
    if ollama_model and processor.ollama_available:
        processor.ollama_model = ollama_model
        logger.info(f"Временно используем модель Ollama: {ollama_model}")
    
    # Создаем временный файл
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file provided")
            
        temp_file.write(content)
        temp_file.close()
        
        logger.info(f"Обработка файла с использованием Ollama: {use_ollama}")
        
        # Обрабатываем файл
        result = processor.process_file(temp_file.name, use_ollama=use_ollama)
        
        # Сохраняем результат
        result_id = str(uuid.uuid4())
        result_filename = f"{result_id}_result.json"
        result_path = os.path.join(RESULTS_DIR, result_filename)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        result['result_id'] = result_id
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Восстанавливаем оригинальную модель
        if ollama_model and processor.ollama_available:
            processor.ollama_model = original_model
        
        # Удаляем временный файл
        if os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass

@app.get("/ollama/models")
async def get_ollama_models():
    """Получение списка доступных моделей Ollama"""
    if not processor or not processor.ollama_available:
        raise HTTPException(status_code=500, detail="Ollama not available")
    
    try:
        import ollama
        client = ollama.Client(host=processor.ollama_host)
        models_response = client.list()
        models = [model['name'] for model in models_response['models']]
        
        return {
            "available_models": models,
            "current_model": processor.ollama_model,
            "ollama_host": processor.ollama_host
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Ollama models: {str(e)}")

@app.post("/ollama/set_model")
async def set_ollama_model(model: str = Query(..., description="Название модели Ollama")):
    """Смена модели Ollama"""
    if not processor or not processor.ollama_available:
        raise HTTPException(status_code=500, detail="Ollama not available")
    
    try:
        import ollama
        client = ollama.Client(host=processor.ollama_host)
        
        # Проверяем, что модель существует
        models_response = client.list()
        available_models = [m['name'] for m in models_response['models']]
        
        if model not in available_models:
            raise HTTPException(status_code=400, detail=f"Model {model} not available. Available models: {available_models}")
        
        processor.ollama_model = model
        
        return {
            "message": f"Model changed to {model}",
            "current_model": processor.ollama_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting Ollama model: {str(e)}")

@app.get("/status")
async def get_status():
    """Получение статуса системы"""
    ollama_status = "available" if processor and processor.ollama_available else "unavailable"
    diarization_status = "available" if processor and processor.diarization_pipeline else "unavailable"
    
    return {
        "processor_available": processor is not None,
        "ollama_status": ollama_status,
        "ollama_model": processor.ollama_model if processor else None,
        "diarization_status": diarization_status,
        "transformers_summarization": processor.transformers_summarization is not None if processor else False
    }

# ... остальные endpoint'ы (results, download, health) остаются без изменений ...
@app.get("/results/{result_id}")
async def get_result(result_id: str):
    result_path = os.path.join(RESULTS_DIR, f"{result_id}_result.json")
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result not found")
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result: {str(e)}")

@app.get("/download/{result_id}")
async def download_result(result_id: str):
    result_path = os.path.join(RESULTS_DIR, f"{result_id}_result.json")
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(
        path=result_path,
        filename=f"transcription_{result_id}.json",
        media_type='application/json'
    )

@app.get("/health")
async def health_check():
    status = "healthy" if processor else "unhealthy"
    return {
        "status": status, 
        "timestamp": datetime.now().isoformat(),
        "processor_initialized": processor is not None,
        "ollama_available": processor.ollama_available if processor else False
    }

@app.get("/")
async def root():
    return {
        "message": "Ollama Audio Processing API v5.0",
        "status": "running" if processor else "initialization failed",
        "features": [
            "Транскрибация аудио/видео",
            "Определение спикеров", 
            "Полный текст без разбивки",
            "Умная суммаризация через Ollama",
            "Резервная суммаризация через трансформеры",
            "Поддержка различных моделей Ollama"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)