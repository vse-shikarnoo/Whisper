import os
import torch
import whisper
import numpy as np
from pyannote.audio import Pipeline
from transformers import pipeline as transformers_pipeline
from pydub import AudioSegment
import tempfile
import librosa
from typing import List, Dict, Tuple, Optional
import json
import logging
from collections import defaultdict
import ollama
import asyncio
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaAudioProcessor:
    def __init__(self, device: str = None, hf_token: str = None, 
                 ollama_model: str = "llama3.2", ollama_host: str = "http://localhost:11434"):
        """
        Инициализация с поддержкой Ollama для суммаризации
        
        Args:
            device: Устройство для вычислений (cuda/cpu)
            hf_token: Токен для Hugging Face
            ollama_model: Модель Ollama для суммаризации
            ollama_host: URL сервера Ollama
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.ollama_available = False
        
        logger.info(f"Используется устройство: {self.device}")
        logger.info(f"Модель Ollama: {ollama_model}")
        
        # Проверяем доступность Ollama
        self._check_ollama()
        
        # Загрузка моделей
        self._load_models()
    
    def _check_ollama(self):
        """Проверка доступности Ollama"""
        try:
            # Пробуем получить список моделей
            client = ollama.Client(host=self.ollama_host)
            models = client.list()
            logger.info(f"Ollama доступен. Модели: {[model['name'] for model in models['models']]}")
            
            # Проверяем, доступна ли выбранная модель
            available_models = [model['name'] for model in models['models']]
            if self.ollama_model not in available_models:
                logger.warning(f"Модель {self.ollama_model} не найдена. Доступные модели: {available_models}")
                # Используем первую доступную модель
                if available_models:
                    self.ollama_model = available_models[0]
                    logger.info(f"Используем модель: {self.ollama_model}")
                else:
                    logger.warning("Нет доступных моделей Ollama")
                    self.ollama_available = False
                    return
            
            self.ollama_available = True
            logger.info(f"Ollama инициализирован с моделью: {self.ollama_model}")
            
        except Exception as e:
            logger.warning(f"Ollama недоступен: {e}. Суммаризация через Ollama отключена.")
            self.ollama_available = False
    
    def _load_models(self):
        """Загрузка всех необходимых моделей"""
        try:
            logger.info("Загрузка модели Whisper для транскрибации...")
            # Используем base модель для скорости, но можно изменить на medium/large
            self.whisper_model = whisper.load_model("base")
            logger.info("Модель Whisper загружена")
            
            # Диаризация
            if self.hf_token:
                logger.info("Загрузка модели для диаризации...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )
                logger.info("Модель диаризации успешно загружена")
            else:
                self.diarization_pipeline = None
                logger.warning("Токен HF не предоставлен, диаризация отключена")
            
            # Трансформеры для суммаризации (резервный вариант)
            logger.info("Загрузка резервной модели для суммаризации...")
            try:
                self.transformers_summarization = transformers_pipeline(
                    "summarization",
                    model="IlyaGusev/rut5_base_sum_gazeta",
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("Резервная модель суммаризации загружена")
            except Exception as e:
                logger.warning(f"Не удалось загрузить резервную модель суммаризации: {e}")
                self.transformers_summarization = None
            
            logger.info("Все модели успешно загружены")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            self.diarization_pipeline = None
            self.transformers_summarization = None
    
    def convert_to_wav(self, input_path: str) -> str:
        """Конвертация в WAV"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            audio, sr = librosa.load(input_path, sr=16000, mono=True)
            import soundfile as sf
            sf.write(temp_path, audio, sr)
            return temp_path
            
        except Exception as e:
            raise Exception(f"Ошибка конвертации: {str(e)}")
    
    def transcribe_audio(self, audio_path: str) -> List[Dict]:
        """Транскрибация аудио"""
        logger.info("Начало транскрибации...")
        
        try:
            # Используем librosa для надежности
            audio_array, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            result = self.whisper_model.transcribe(
                audio_array,
                language="ru",
                task="transcribe",
                verbose=False
            )
            
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "speaker": None
                })
            
            logger.info(f"Транскрибация завершена. Сегментов: {len(segments)}")
            return segments
            
        except Exception as e:
            logger.error(f"Ошибка транскрибации: {e}")
            raise
    
    def get_full_text(self, segments: List[Dict]) -> str:
        """Получение полного текста без разбивки"""
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        full_text = " ".join(segment["text"] for segment in sorted_segments)
        full_text = " ".join(full_text.split())
        return full_text
    
    def diarize_audio(self, audio_path: str) -> List[Dict]:
        """Диаризация аудио"""
        if not self.diarization_pipeline:
            return []
        
        logger.info("Начало диаризации...")
        
        try:
            diarization = self.diarization_pipeline(audio_path)
            
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            logger.info(f"Диаризация завершена. Сегментов спикеров: {len(speaker_segments)}")
            return speaker_segments
            
        except Exception as e:
            logger.error(f"Ошибка диаризации: {e}")
            return []
    
    def assign_speakers(self, transcription_segments: List[Dict], speaker_segments: List[Dict]) -> List[Dict]:
        """Сопоставление спикеров с сегментами"""
        if not speaker_segments:
            for i, segment in enumerate(transcription_segments):
                segment["speaker"] = f"speaker_{i % 2 + 1}"
            return transcription_segments
        
        for trans_segment in transcription_segments:
            trans_start = trans_segment["start"]
            trans_end = trans_segment["end"]
            
            best_speaker = None
            max_overlap = 0
            
            for speaker_segment in speaker_segments:
                speaker_start = speaker_segment["start"]
                speaker_end = speaker_segment["end"]
                
                overlap_start = max(trans_start, speaker_start)
                overlap_end = min(trans_end, speaker_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker_segment["speaker"]
            
            trans_segment["speaker"] = best_speaker if best_speaker else "unknown"
        
        return transcription_segments
    
    def summarize_with_ollama(self, text: str, context: str = "") -> str:
        """
        Суммаризация текста с помощью Ollama
        
        Args:
            text: Текст для суммаризации
            context: Контекст (например, "речь спикера 1")
        """
        if not self.ollama_available:
            return self.summarize_with_transformers(text)
        
        try:
            client = ollama.Client(host=self.ollama_host)
            
            # Подготавливаем промпт в зависимости от контекста
            if context:
                prompt = f"""Проанализируй следующий текст и создай качественное краткое изложение, сохраняя основные идеи и смысл:

{text}

Контекст: {context}

Требования к изложению:
- Сохрани ключевые идеи и основные выводы
- Используй ясный и понятный язык
- Избегай повторений
- Выдели наиболее важные моменты
- Объем: примерно 10-15% от исходного текста

Краткое изложение:"""
            else:
                prompt = f"""Создай качественное краткое изложение следующего текста, сохраняя основные идеи и смысл:

{text}

Требования к изложению:
- Выдели главные темы и ключевые моменты
- Сохрани структуру и логику изложения
- Используй ясный и связный язык
- Избегай потери важной информации
- Объем: примерно 10-15% от исходного текста

Краткое изложение:"""
            
            # Ограничиваем длину текста для моделей с ограниченным контекстом
            words = text.split()
            if len(words) > 2000:
                text = " ".join(words[:2000])
                logger.warning("Текст сокращен для суммаризации")
            
            response = client.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'num_predict': 500
                }
            )
            
            summary = response['response'].strip()
            
            # Очищаем ответ от возможных артефактов
            summary = re.sub(r'^(Краткое изложение|Изложение|Резюме|Summary)[:\s]*', '', summary, flags=re.IGNORECASE)
            summary = summary.strip()
            
            if not summary or len(summary) < 50:
                logger.warning("Ollama вернул слишком короткую суммаризацию, используем резервный метод")
                return self.summarize_with_transformers(text)
            
            logger.info(f"Суммаризация через Ollama завершена. Длина: {len(summary)} символов")
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка суммаризации через Ollama: {e}")
            return self.summarize_with_transformers(text)
    
    def summarize_with_transformers(self, text: str) -> str:
        """Резервная суммаризация через трансформеры"""
        if not self.transformers_summarization:
            # Простая эвристическая суммаризация
            sentences = text.split('. ')
            if len(sentences) <= 3:
                return text
            
            important_sentences = sentences[:2] + sentences[-2:]
            return '. '.join(important_sentences) + '.'
        
        try:
            words = text.split()
            if len(words) > 512:
                text = " ".join(words[:512])
            
            summary = self.transformers_summarization(
                text,
                max_length=150,
                min_length=50,
                do_sample=False
            )[0]['summary_text']
            return summary
        except Exception as e:
            logger.warning(f"Ошибка резервной суммаризации: {e}")
            return text[:300] + "..." if len(text) > 300 else text
    
    def smart_summarize(self, text: str, context: str = "") -> str:
        """Умная суммаризация с выбором лучшего метода"""
        if self.ollama_available:
            return self.summarize_with_ollama(text, context)
        else:
            return self.summarize_with_transformers(text)
    
    def process_file(self, file_path: str, use_ollama: bool = True) -> Dict:
        """Основной метод обработки файла"""
        logger.info(f"Обработка файла: {file_path}")
        
        wav_path = self.convert_to_wav(file_path)
        temp_file_created = wav_path != file_path
        
        try:
            # Транскрибация
            transcription_segments = self.transcribe_audio(wav_path)
            
            # Полный текст
            full_text = self.get_full_text(transcription_segments)
            
            # Суммаризация полного текста
            if use_ollama and self.ollama_available:
                full_text_summary = self.summarize_with_ollama(full_text, "полная транскрибация")
            else:
                full_text_summary = self.smart_summarize(full_text)
            
            # Диаризация
            speaker_segments = self.diarize_audio(wav_path)
            
            # Сопоставление спикеров
            segments_with_speakers = self.assign_speakers(transcription_segments, speaker_segments)
            
            # Группировка по спикерам
            speaker_texts = defaultdict(list)
            for segment in segments_with_speakers:
                speaker_texts[segment["speaker"]].append(segment["text"])
            
            speaker_texts = {speaker: " ".join(texts) for speaker, texts in speaker_texts.items()}
            
            # Суммаризация по спикерам
            speaker_summaries = {}
            for speaker, text in speaker_texts.items():
                if use_ollama and self.ollama_available:
                    speaker_summaries[speaker] = self.summarize_with_ollama(text, f"речь {speaker}")
                else:
                    speaker_summaries[speaker] = self.smart_summarize(text)
            
            result = {
                # Основные данные
                "full_text": full_text,
                "full_text_summary": full_text_summary,
                
                # Детализированные данные
                "segments": segments_with_speakers,
                "speaker_texts": speaker_texts,
                "speaker_summaries": speaker_summaries,
                
                # Статистика
                "speakers_count": len(speaker_texts),
                "total_segments": len(segments_with_speakers),
                "total_text_length": len(full_text),
                
                # Информация о методах
                "diarization_enabled": self.diarization_pipeline is not None,
                "ollama_enabled": use_ollama and self.ollama_available,
                "ollama_model": self.ollama_model if use_ollama and self.ollama_available else None,
                "summarization_method": "ollama" if use_ollama and self.ollama_available else "transformers",
                
                # Качество обработки
                "processing_quality": "high" if (self.diarization_pipeline and use_ollama and self.ollama_available) else "medium"
            }
            
            logger.info("Обработка файла завершена успешно")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка обработки файла: {e}")
            raise
        finally:
            if temp_file_created and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except:
                    pass