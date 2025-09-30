# Audio Transcription & Summarization API

Local application for audio/video file transcription with speaker identification and intelligent text summarization.

## 🌟 Features

- 🎤 **Transcription** of audio and video files (Whisper)
- 👥 **Speaker identification** (diarization)
- 📝 **Full text** without segmentation
- 🤖 **Intelligent summarization** via Ollama
- 🔄 **Fallback summarization** via transformers
- 📊 **Detailed segments** with timestamps
- 🚀 **Fully local operation**
- 🔧 **REST API** for integration

## 🛠 Technologies

- **Whisper** - audio transcription
- **PyAnnote** - speaker diarization
- **Ollama** - intelligent summarization
- **Transformers** - fallback summarization
- **FastAPI** - web interface
- **PyTorch** - machine learning

## 📦 Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd audio-transcription-api
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama (Optional)

**Windows**: Download from [ollama.com](https://ollama.com/)

**Linux/Mac**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 4. Environment Variables

```bash
# For diarization (optional)
export HF_TOKEN="your_huggingface_token"

# For Ollama (optional)
export OLLAMA_MODEL="llama3.2"
export OLLAMA_HOST="http://localhost:11434"
```

## 🚀 Quick Start

```bash
python main.py
```

Server will start at `http://localhost:8000`

## 📚 API Documentation

### Main Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/process` | Process audio/video file |
| `GET` | `/results/{id}` | Get result by ID |
| `GET` | `/download/{id}` | Download result as JSON |
| `GET` | `/health` | API health check |
| `GET` | `/status` | Detailed system status |
| `GET` | `/ollama/models` | List Ollama models |

### File Processing

#### Basic Request

```bash
curl -X POST "http://localhost:8000/process" \
     -F "file=@audio.mp3"
```

#### With Parameters

```bash
# With Ollama (recommended)
curl -X POST "http://localhost:8000/process?use_ollama=true" \
     -F "file=@audio.mp3"

# Without Ollama (uses transformers)
curl -X POST "http://localhost:8000/process?use_ollama=false" \
     -F "file=@audio.mp3"

# Specific Ollama model
curl -X POST "http://localhost:8000/process?use_ollama=true&ollama_model=mistral" \
     -F "file=@audio.mp3"
```

### Getting Results

```bash
# Get result by ID
curl "http://localhost:8000/results/12345678-1234-1234-1234-123456789abc"

# Download as file
curl -o transcription.json \
     "http://localhost:8000/download/12345678-1234-1234-1234-123456789abc"
```

### Ollama Management

```bash
# Get available models
curl "http://localhost:8000/ollama/models"

# Change Ollama model
curl -X POST "http://localhost:8000/ollama/set_model?model=llama3.2"
```

## 📋 Supported Formats

### Audio
- MP3, WAV, M4A, FLAC

### Video
- MP4, AVI, MOV, MKV

## 📊 Response Structure

```json
{
  "result_id": "uuid",
  "full_text": "Complete transcription text without segmentation...",
  "full_text_summary": "Summary of the entire text...",
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Segment text",
      "speaker": "speaker_1"
    }
  ],
  "speaker_texts": {
    "speaker_1": "All text from speaker 1...",
    "speaker_2": "All text from speaker 2..."
  },
  "speaker_summaries": {
    "speaker_1": "Summary of speaker 1's speech...",
    "speaker_2": "Summary of speaker 2's speech..."
  },
  "speakers_count": 2,
  "total_segments": 15,
  "total_text_length": 2450,
  "ollama_enabled": true,
  "ollama_model": "llama3.2",
  "summarization_method": "ollama",
  "processing_quality": "high"
}
```

## 🎯 Usage Examples

### Python Client

```python
import requests

def process_audio(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'audio/mpeg')}
        response = requests.post(
            'http://localhost:8000/process?use_ollama=true',
            files=files
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Processing complete! ID: {result['result_id']}")
        print(f"Speakers: {result['speakers_count']}")
        print(f"Summary: {result['full_text_summary']}")
        return result
    else:
        print(f"Error: {response.text}")
        return None

# Usage
result = process_audio('meeting.mp3')
```

### Bash Script

```bash
#!/bin/bash
process_audio() {
    local file=$1
    local response=$(curl -s -X POST "http://localhost:8000/process?use_ollama=true" -F "file=@$file")
    local result_id=$(echo $response | grep -o '"result_id":"[^"]*"' | cut -d'"' -f4)
    
    if [ -n "$result_id" ]; then
        echo "Processing complete. ID: $result_id"
        curl -s "http://localhost:8000/results/$result_id" | jq '.full_text_summary'
    else
        echo "Processing error"
        echo $response
    fi
}

process_audio $1
```

## ⚙️ Model Configuration

### Whisper Models
- `tiny` - fast, low quality
- `base` - balanced speed/quality
- `small` - good quality
- `medium` - high quality (recommended)
- `large` - best quality, slow

Change in code:
```python
self.whisper_model = whisper.load_model("medium")
```

### Ollama Models
```bash
# Install models
ollama pull llama3.2
ollama pull mistral
ollama pull codellama

# Check installed models
ollama list
```

## 🔧 Troubleshooting

### Diarization Issues
```bash
# If no HF_TOKEN, diarization will be disabled
export HF_TOKEN="your_huggingface_token"
```

### Ollama Issues
```bash
# Check Ollama status
ollama serve

# Ensure models are loaded
curl http://localhost:11434/api/tags
```

### Dependency Issues
```bash
# Reinstall PyTorch (for GPU)
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install FFmpeg
# Ubuntu/Debian
sudo apt install ffmpeg

# Windows: download from ffmpeg.org
```

## 📈 Performance

| Component | Resources | Processing Time (1 min audio) |
|-----------|-----------|-------------------------------|
| Whisper Base | 1GB RAM | ~30 seconds |
| Whisper Medium | 2GB RAM | ~60 seconds |
| Diarization | 2GB RAM | ~45 seconds |
| Ollama Summarization | 4GB RAM | ~15 seconds |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

## 📞 Support

If you encounter issues:

1. Check all dependencies are installed
2. Ensure Ollama is running (`ollama serve`)
3. Check application logs
4. Create an issue in the repository
