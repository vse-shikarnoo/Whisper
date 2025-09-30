from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
import tempfile
import json
import uuid
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Audio Processing API", version="4.0.0")

# Пробуем загрузить улучшенный процессор
try:
    from enhanced_audio_processor import EnhancedAudioProcessor
    from simple_diarization import SimpleDiarization
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    enhanced_processor = EnhancedAudioProcessor(hf_token=HF_TOKEN)
    simple_diarization = SimpleDiarization()
    
    logger.info("Улучшенный процессор загружен")
except Exception as e:
    logger.error(f"Ошибка загрузки улучшенного процессора: {e}")
    enhanced_processor = None
    simple_diarization = None

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/process")
async def process_audio_file(file: UploadFile = File(...), method: str = "enhanced"):
    """
    Обработка аудио/видео файла с выбором метода диаризации
    
    Parameters:
    - method: "enhanced" (улучшенная), "simple" (простая), "auto" (автовыбор)
    """
    if not enhanced_processor and method != "simple":
        raise HTTPException(status_code=500, detail="Enhanced processor not available")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Проверяем расширение файла
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.mp4', '.avi', '.mov', '.mkv'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="File type not supported")
    
    # Создаем временный файл
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file provided")
            
        temp_file.write(content)
        temp_file.close()
        
        logger.info(f"Обработка файла методом: {method}")
        
        # Выбираем метод обработки
        if method == "simple":
            result = await process_simple_method(temp_file.name)
        elif method == "enhanced":
            result = enhanced_processor.process_file(temp_file.name)
        else:  # auto
            try:
                result = enhanced_processor.process_file(temp_file.name)
            except Exception as e:
                logger.warning(f"Улучшенный метод не сработал: {e}")
                result = await process_simple_method(temp_file.name)
        
        # Сохраняем результат
        result_id = str(uuid.uuid4())
        result_filename = f"{result_id}_result.json"
        result_path = os.path.join(RESULTS_DIR, result_filename)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        result['result_id'] = result_id
        result['processing_method'] = method
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        if os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass

async def process_simple_method(file_path: str):
    """Простой метод обработки с базовой диаризацией"""
    import whisper
    
    # Базовая транскрибация
    model = whisper.load_model("base")
    result = model.transcribe(file_path, language="ru", verbose=False)
    
    segments = []
    for segment in result["segments"]:
        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip(),
            "speaker": None
        })
    
    # ПОЛУЧАЕМ ПОЛНЫЙ ТЕКСТ
    full_text = result["text"]
    
    # ПРОСТАЯ СУММАРИЗАЦИЯ ПОЛНОГО ТЕКСТА
    full_text_summary = full_text[:500] + "..." if len(full_text) > 500 else full_text
    
    # Простая диаризация
    if simple_diarization:
        segments = simple_diarization.diarize_by_silence(segments)
        segments = simple_diarization.merge_short_segments(segments)
    else:
        # Базовая эвристика
        for i, segment in enumerate(segments):
            segment["speaker"] = f"speaker_{i % 2 + 1}"
    
    # Группировка по спикерам
    speaker_texts = {}
    for segment in segments:
        speaker = segment["speaker"]
        if speaker not in speaker_texts:
            speaker_texts[speaker] = []
        speaker_texts[speaker].append(segment["text"])
    
    speaker_texts = {k: " ".join(v) for k, v in speaker_texts.items()}
    
    # Суммаризация по спикерам
    speaker_summaries = {}
    for speaker, text in speaker_texts.items():
        speaker_summaries[speaker] = text[:200] + "..." if len(text) > 200 else text
    
    return {
        # Полный текст и его суммаризация
        "full_text": full_text,
        "full_text_summary": full_text_summary,
        
        # Детализированные данные
        "segments": segments,
        "speaker_texts": speaker_texts,
        "speaker_summaries": speaker_summaries,
        
        # Статистика
        "speakers_count": len(speaker_texts),
        "total_segments": len(segments),
        "total_text_length": len(full_text),
        "diarization_quality": "simple",
        "speakers_detected": list(speaker_texts.keys()),
        
        # Метаинформация
        "processing_method": "simple",
        "summarization_enabled": False
    }

@app.get("/methods")
async def get_available_methods():
    """Доступные методы обработки"""
    return {
        "methods": [
            {
                "name": "enhanced",
                "description": "Улучшенная диаризация с pyannote (требуется HF_TOKEN)",
                "available": enhanced_processor is not None and enhanced_processor.diarization_pipeline is not None
            },
            {
                "name": "simple", 
                "description": "Простая диаризация на основе пауз",
                "available": simple_diarization is not None
            },
            {
                "name": "auto",
                "description": "Автоматический выбор лучшего доступного метода",
                "available": enhanced_processor is not None or simple_diarization is not None
            }
        ]
    }

@app.get("/results/{result_id}")
async def get_result(result_id: str):
    """Получение результата по ID"""
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
    """Скачивание результата в виде JSON файла"""
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
    """Проверка статуса API"""
    status = "healthy" if (enhanced_processor or simple_diarization) else "unhealthy"
    return {
        "status": status, 
        "timestamp": datetime.now().isoformat(),
        "enhanced_processor": enhanced_processor is not None,
        "simple_diarization": simple_diarization is not None
    }

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "Enhanced Audio Processing API v4.0",
        "status": "running" if (enhanced_processor or simple_diarization) else "initialization failed",
        "features": [
            "Транскрибация аудио/видео",
            "Определение спикеров", 
            "Полный текст без разбивки",
            "Суммаризация полного текста",
            "Суммаризация по спикерам",
            "Детализированные сегменты"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)