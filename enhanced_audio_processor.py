import os
import torch
import whisper
import numpy as np
from pyannote.audio import Pipeline
from transformers import pipeline as transformers_pipeline
from pydub import AudioSegment
import tempfile
import librosa
from typing import List, Dict, Tuple
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAudioProcessor:
    def __init__(self, device: str = None, hf_token: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token
        
        logger.info(f"Используется устройство: {self.device}")
        self._load_models()
    
    def _load_models(self):
        """Загрузка моделей с улучшенными настройками"""
        try:
            logger.info("Загрузка модели Whisper для транскрибации...")
            # Используем medium модель для лучшего качества
            self.whisper_model = whisper.load_model("medium")
            
            # Диаризация с улучшенными настройками
            if self.hf_token:
                logger.info("Загрузка улучшенной модели для диаризации...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )
                # Настройка параметров диаризации для лучшего качества
                self.diarization_pipeline.instantiate({
                    "min_duration_on": 0.5,  # Минимальная длительность сегмента
                    "min_duration_off": 0.5,  # Минимальная пауза между сегментами
                })
            else:
                self.diarization_pipeline = None
                logger.warning("Токен HF не предоставлен, диаризация отключена")
            
            # Суммаризация
            logger.info("Загрузка модели для суммаризации...")
            try:
                self.summarization_pipeline = transformers_pipeline(
                    "summarization",
                    model="IlyaGusev/rut5_base_sum_gazeta",
                    device=0 if self.device == "cuda" else -1
                )
            except:
                self.summarization_pipeline = None
                logger.warning("Суммаризация отключена")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            self.diarization_pipeline = None
            self.summarization_pipeline = None
    
    def convert_to_wav(self, input_path: str) -> str:
        """Конвертация в WAV с оптимизацией для диаризации"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Используем оптимальные настройки для диаризации
            audio, sr = librosa.load(input_path, sr=16000, mono=True)
            import soundfile as sf
            sf.write(temp_path, audio, sr)
            return temp_path
            
        except Exception as e:
            raise Exception(f"Ошибка конвертации: {str(e)}")
    
    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict]:
        """Транскрибация с детальными временными метками"""
        logger.info("Транскрибация с детальными метками...")
        
        try:
            # Используем word_timestamps для более точного сопоставления
            result = self.whisper_model.transcribe(
                audio_path,
                language="ru",
                task="transcribe",
                word_timestamps=True,  # Включить временные метки слов
                verbose=False
            )
            
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "words": segment.get("words", []),  # Сохраняем слова с временными метками
                    "speaker": None
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"Ошибка транскрибации: {e}")
            # Fallback без word_timestamps
            result = self.whisper_model.transcribe(
                audio_path,
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
                    "words": [],
                    "speaker": None
                })
            
            return segments
    
    def get_full_text(self, segments: List[Dict]) -> str:
        """Получение полного текста без разбивки по сегментам и спикерам"""
        # Сортируем сегменты по времени на случай если они не в порядке
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        
        # Собираем полный текст
        full_text = " ".join(segment["text"] for segment in sorted_segments)
        
        # Очищаем текст от лишних пробелов
        full_text = " ".join(full_text.split())
        
        return full_text
    
    def enhanced_diarization(self, audio_path: str) -> List[Dict]:
        """Улучшенная диаризация с постобработкой"""
        if not self.diarization_pipeline:
            return []
        
        logger.info("Запуск улучшенной диаризации...")
        
        try:
            # Выполняем диаризацию
            diarization = self.diarization_pipeline(audio_path)
            
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            # Постобработка: объединяем очень короткие сегменты
            speaker_segments = self._merge_short_segments(speaker_segments)
            
            logger.info(f"Диаризация завершена. Найдено {len(speaker_segments)} сегментов")
            return speaker_segments
            
        except Exception as e:
            logger.error(f"Ошибка диаризации: {e}")
            return []
    
    def _merge_short_segments(self, segments: List[Dict], min_duration: float = 1.0) -> List[Dict]:
        """Объединение коротких сегментов одного спикера"""
        if not segments:
            return segments
        
        merged = []
        current_segment = segments[0].copy()
        
        for segment in segments[1:]:
            # Если тот же спикер и небольшой промежуток
            if (segment["speaker"] == current_segment["speaker"] and 
                segment["start"] - current_segment["end"] < 2.0):  # Промежуток менее 2 секунд
                current_segment["end"] = segment["end"]
            else:
                # Проверяем длительность текущего сегмента
                duration = current_segment["end"] - current_segment["start"]
                if duration >= min_duration:
                    merged.append(current_segment)
                current_segment = segment.copy()
        
        # Добавляем последний сегмент
        duration = current_segment["end"] - current_segment["start"]
        if duration >= min_duration:
            merged.append(current_segment)
        
        return merged
    
    def smart_speaker_assignment(self, transcription_segments: List[Dict], 
                               speaker_segments: List[Dict]) -> List[Dict]:
        """Умное сопоставление спикеров с использованием временных меток слов"""
        logger.info("Умное сопоставление спикеров...")
        
        if not speaker_segments:
            logger.warning("Нет данных о спикерах. Используем улучшенную эвристику.")
            return self._advanced_speaker_heuristic(transcription_segments)
        
        # Создаем временную шкалу активности спикеров
        speaker_activity = self._create_speaker_activity_map(speaker_segments)
        
        for trans_segment in transcription_segments:
            # Используем слова для более точного определения спикера
            if trans_segment.get("words"):
                speaker = self._assign_speaker_by_words(trans_segment, speaker_activity)
            else:
                speaker = self._assign_speaker_by_segment(trans_segment, speaker_activity)
            
            trans_segment["speaker"] = speaker
        
        return transcription_segments
    
    def _create_speaker_activity_map(self, speaker_segments: List[Dict]) -> Dict[float, str]:
        """Создание карты активности спикеров во времени"""
        activity = {}
        time_points = set()
        
        for segment in speaker_segments:
            time_points.add(segment["start"])
            time_points.add(segment["end"])
        
        sorted_times = sorted(time_points)
        
        # Для каждого интервала определяем доминирующего спикера
        for i in range(len(sorted_times) - 1):
            start = sorted_times[i]
            end = sorted_times[i + 1]
            
            # Находим спикера, который говорит большую часть этого интервала
            speakers_duration = defaultdict(float)
            for segment in speaker_segments:
                overlap_start = max(start, segment["start"])
                overlap_end = min(end, segment["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > 0:
                    speakers_duration[segment["speaker"]] += overlap_duration
            
            if speakers_duration:
                dominant_speaker = max(speakers_duration, key=speakers_duration.get)
                activity[(start + end) / 2] = dominant_speaker  # Средняя точка интервала
        
        return activity
    
    def _assign_speaker_by_words(self, trans_segment: Dict, speaker_activity: Dict) -> str:
        """Определение спикера на основе временных меток слов"""
        word_speakers = []
        
        for word_info in trans_segment["words"]:
            word_time = (word_info["start"] + word_info["end"]) / 2
            
            # Находим ближайшую временную точку в активности спикеров
            closest_time = min(speaker_activity.keys(), 
                             key=lambda x: abs(x - word_time), 
                             default=None)
            
            if closest_time and abs(closest_time - word_time) < 5.0:  # В пределах 5 секунд
                word_speakers.append(speaker_activity[closest_time])
        
        # Выбираем наиболее частого спикера для слов в сегменте
        if word_speakers:
            from collections import Counter
            speaker_counts = Counter(word_speakers)
            return speaker_counts.most_common(1)[0][0]
        else:
            return self._assign_speaker_by_segment(trans_segment, speaker_activity)
    
    def _assign_speaker_by_segment(self, trans_segment: Dict, speaker_activity: Dict) -> str:
        """Определение спикера для всего сегмента"""
        segment_mid = (trans_segment["start"] + trans_segment["end"]) / 2
        
        # Находим ближайшую временную точку
        closest_time = min(speaker_activity.keys(), 
                         key=lambda x: abs(x - segment_mid), 
                         default=None)
        
        if closest_time and abs(closest_time - segment_mid) < 10.0:  # В пределах 10 секунд
            return speaker_activity[closest_time]
        else:
            return "unknown"
    
    def _advanced_speaker_heuristic(self, segments: List[Dict]) -> List[Dict]:
        """Улучшенная эвристика для определения спикеров"""
        # Анализируем паузы между сегментами для определения смены спикера
        speakers = []
        current_speaker = "speaker_1"
        
        for i, segment in enumerate(segments):
            if i > 0:
                # Большая пауза может означать смену спикера
                pause = segment["start"] - segments[i-1]["end"]
                if pause > 5.0:  # Пауза более 5 секунд
                    if current_speaker == "speaker_1":
                        current_speaker = "speaker_2"
                    else:
                        current_speaker = "speaker_1"
            
            segment["speaker"] = current_speaker
            speakers.append(current_speaker)
        
        # Если нашли только одного спикера, проверяем разнообразие текста
        unique_speakers = set(speakers)
        if len(unique_speakers) == 1 and len(segments) > 5:
            # Пытаемся найти естественные точки раздела
            for i in range(2, len(segments)-2):
                # Если есть большая пауза в середине
                if (segments[i]["start"] - segments[i-1]["end"] > 3.0 and
                    segments[i+1]["start"] - segments[i]["end"] > 3.0):
                    for j in range(i, len(segments)):
                        segments[j]["speaker"] = "speaker_2"
                    break
        
        return segments
    
    def summarize_full_text(self, text: str, max_length: int = 200) -> str:
        """Суммаризация полного текста"""
        if not self.summarization_pipeline:
            # Простая эвристическая суммаризация
            sentences = text.split('. ')
            if len(sentences) <= 3:
                return text
            
            # Берем первые и последние предложения
            important_sentences = sentences[:2] + sentences[-1:]
            return '. '.join(important_sentences) + '.'
        
        try:
            # Ограничиваем длину текста для модели
            words = text.split()
            if len(words) > 1024:
                text = " ".join(words[:1024])
            
            summary = self.summarization_pipeline(
                text,
                max_length=max_length,
                min_length=50,
                do_sample=False
            )[0]['summary_text']
            return summary
        except Exception as e:
            logger.warning(f"Ошибка суммаризации полного текста: {e}")
            # Fallback - первые 300 символов
            return text[:300] + "..." if len(text) > 300 else text
    
    def process_file(self, file_path: str) -> Dict:
        """Основной метод обработки с улучшенной диаризацией"""
        logger.info(f"Обработка файла с улучшенной диаризацией: {file_path}")
        
        wav_path = self.convert_to_wav(file_path)
        temp_file_created = wav_path != file_path
        
        try:
            # Транскрибация с детальными метками
            transcription_segments = self.transcribe_with_timestamps(wav_path)
            
            # ПОЛУЧАЕМ ПОЛНЫЙ ТЕКСТ БЕЗ РАЗБИВКИ
            full_text = self.get_full_text(transcription_segments)
            
            # СУММАРИЗАЦИЯ ПОЛНОГО ТЕКСТА
            full_text_summary = self.summarize_full_text(full_text)
            
            # Улучшенная диаризация
            speaker_segments = self.enhanced_diarization(wav_path)
            
            # Умное сопоставление
            segments_with_speakers = self.smart_speaker_assignment(
                transcription_segments, speaker_segments
            )
            
            # Группировка по спикерам
            speaker_texts = self._group_text_by_speakers(segments_with_speakers)
            
            # Суммаризация по спикерам (если нужно)
            speaker_summaries = {}
            for speaker, text in speaker_texts.items():
                speaker_summaries[speaker] = self.summarize_full_text(text, max_length=100)
            
            result = {
                # Полный текст и его суммаризация
                "full_text": full_text,
                "full_text_summary": full_text_summary,
                
                # Детализированные данные по сегментам и спикерам
                "segments": segments_with_speakers,
                "speaker_texts": speaker_texts,
                "speaker_summaries": speaker_summaries,
                
                # Статистика
                "speakers_count": len(speaker_texts),
                "total_segments": len(segments_with_speakers),
                "total_text_length": len(full_text),
                "diarization_quality": "enhanced",
                "speakers_detected": list(speaker_texts.keys()),
                
                # Метаинформация
                "processing_method": "enhanced",
                "summarization_enabled": self.summarization_pipeline is not None
            }
            
            logger.info("Улучшенная обработка завершена")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка улучшенной обработки: {e}")
            raise
        finally:
            if temp_file_created and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except:
                    pass
    
    def _group_text_by_speakers(self, segments: List[Dict]) -> Dict[str, str]:
        """Группировка текста по спикерам"""
        speaker_texts = defaultdict(list)
        
        for segment in segments:
            speaker_texts[segment["speaker"]].append(segment["text"])
        
        return {speaker: " ".join(texts) for speaker, texts in speaker_texts.items()}