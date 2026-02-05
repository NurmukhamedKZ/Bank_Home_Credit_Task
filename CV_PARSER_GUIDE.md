# 🚀 CV Parser Quick Guide

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install scikit-learn numpy

# 2. Test the parser
cd app/service
python test_parser.py

# 3. Process all CVs
cd ../..
python app/process_cvs.py
```

## 📖 Usage

```python
from app.service import CVParser

# Initialize
parser = CVParser(collection_name="CVs")

# Process a CV
result = parser.process_cv("data/CVs/resume.pdf")

print(f"✅ {result['full_name']} processed!")
```

## 🔑 Key Changes

✅ **Replaced BGE-M3 with TF-IDF**
- 10-50x faster sparse embeddings
- 40x less memory usage
- No heavy model downloads

## 📚 Documentation

- **Full docs**: `app/service/README.md`
- **Quick start**: `app/service/QUICKSTART.md`
- **TF-IDF comparison**: `app/service/TFIDF_vs_BGE.md`
- **Examples**: `app/service/example_usage.py`
- **Tests**: `app/service/test_parser.py`

## 🎯 What It Does

1. **Parse** PDF/DOCX/TXT files → raw text
2. **Extract** structured data via GPT-4o-mini → JSON
3. **Save files**:
   - `data/Raw_CVs/*.txt` - Raw extracted text
   - `data/CV_JSONs/*.json` - Structured data
4. **Embed** with Voyage AI (dense) + TF-IDF (sparse)
5. **Save** to Qdrant vector database

## 📊 Output Structure

```json
{
    "full_name": "John Doe",
    "email": "john@example.com",
    "phone": "+1234567890",
    "skills": ["Python", "FastAPI", "Docker"],
    "total_experience_months": 36,
    "work_history": [...],
    "education": [...]
}
```

## 🔍 Search Example

```python
# Search for candidates
results = parser.qdrant_client.query_points(
    collection_name="CVs",
    query=parser.dense_model.embed_documents(["Python developer"])[0],
    limit=5
)

for point in results.points:
    print(f"{point.payload['full_name']} - {point.score:.4f}")
```

## ⚙️ Configuration

Edit `.env` file:
```env
LLAMA_PARSE_API=your_key
OPENAI_API_KEY=your_key
QDRANT_API=your_key
QDRANT_URL=your_url
VOYAGE_API=your_key
```

## 🆘 Need Help?

1. Run tests: `python app/service/test_parser.py`
2. Check logs for errors
3. See `app/service/README.md` for detailed docs

## 📦 Project Structure

```
Bank_Home_Credit_Task/
├── app/
│   ├── service/
│   │   ├── Parse_pdf.py          # Main parser class
│   │   ├── example_usage.py      # Usage examples
│   │   └── test_parser.py        # Tests
│   └── process_cvs.py            # Batch processing script
├── data/
│   └── CVs/                      # Put CV files here
└── research/
    └── CV_parser.ipynb           # Original research notebook
```

Ready to go! 🎉
