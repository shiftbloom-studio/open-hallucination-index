# Wikipedia Ingestion Pipeline

High-performance Wikipedia ingestion pipeline for the Open Hallucination Index project.

## Features

- **10-50x faster** than the original monolithic script
- **Producer-consumer architecture** with non-blocking queues
- **Parallel downloads** with resume support for Wikipedia dumps
- **GPU-accelerated embeddings** with large batch sizes (512)
- **Async uploads** to both Qdrant and Neo4j
- **10+ relationship types** in Neo4j knowledge graph
- **Resumable checkpoints** for crash recovery
- **Real-time progress** with rich statistics

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  Download   │───▶│  Preprocess  │───▶│   Embed     │───▶│   Upload    │
│  (4 threads)│    │  (8 threads) │    │  (GPU batch)│    │  (8 threads)│
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
      │                  │                   │                   │
      ▼                  ▼                   ▼                   ▼
   Wikipedia         Text Clean          GPU Encode        Qdrant + Neo4j
    Dumps            + Chunking          + BM25 Sparse     Parallel Upload
```

## Quick Start

```bash
# Run from project root
python -m ingestion --help

# Basic usage with defaults
python -m ingestion \
    --wiki-dump ~/downloads/enwiki-pages-articles.xml.bz2 \
    --qdrant-url http://localhost:6333 \
    --neo4j-uri bolt://localhost:7687

# High-performance settings
python -m ingestion \
    --wiki-dump ~/downloads/enwiki-pages-articles.xml.bz2 \
    --batch-size 512 \
    --max-articles 1000000 \
    --download-workers 4 \
    --preprocess-workers 8 \
    --upload-workers 8 \
    --gpu
```

## Module Structure

| Module | Description |
|--------|-------------|
| `models.py` | Data classes: `WikiArticle`, `ProcessedChunk`, `IngestionConfig` |
| `downloader.py` | Parallel Wikipedia dump downloading with resume |
| `preprocessor.py` | Text cleaning, chunking, BM25 tokenization |
| `qdrant_store.py` | Async vector store with GPU embeddings |
| `neo4j_store.py` | Graph store with 10+ relationship types |
| `checkpoint.py` | Resumable ingestion state management |
| `pipeline.py` | Main producer-consumer orchestration |
| `__main__.py` | CLI entry point |

## Neo4j Relationship Types

The pipeline creates rich relationships between articles:

| Relationship | Description |
|--------------|-------------|
| `LINKS_TO` | Internal wiki links between articles |
| `IN_CATEGORY` | Article belongs to category |
| `MENTIONS` | Article mentions entity |
| `SEE_ALSO` | Explicit "See also" references |
| `DISAMBIGUATES` | Disambiguation page links |
| `LOCATED_IN` | Geographic location relationships |
| `HAS_OCCUPATION` | Person's occupation |
| `HAS_NATIONALITY` | Person's nationality |
| `RELATED_TO` | Category co-occurrence relationships |
| `NEXT` | Section ordering within article |

## Configuration Options

### Performance Tuning

```python
IngestionConfig(
    batch_size=512,           # GPU embedding batch size
    chunk_size=400,           # Words per chunk
    chunk_overlap=50,         # Overlap between chunks
    download_workers=4,       # Parallel download threads
    preprocess_workers=8,     # Text processing threads
    upload_workers=4,         # Upload threads per store
    download_queue_size=8,    # Pending downloads
    preprocess_queue_size=2048,  # Pending chunks
    upload_queue_size=16,     # Pending uploads
)
```

### Database Settings

```python
IngestionConfig(
    qdrant_url="http://localhost:6333",
    qdrant_collection="wiki_articles",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    neo4j_database="neo4j",
)
```

## Requirements

```txt
sentence-transformers>=2.2.0
qdrant-client>=1.6.0
neo4j>=5.0.0
lxml>=4.9.0
mwparserfromhell>=0.6.0
nltk>=3.8.0
tqdm>=4.65.0
```

## Comparison with Original Script

| Metric | Original | New Pipeline | Improvement |
|--------|----------|--------------|-------------|
| Articles/sec | ~1 | 10-50 | 10-50x |
| CPU utilization | ~10% | ~80% | 8x |
| GPU utilization | 0% | ~90% | Full usage |
| Memory efficiency | High | Streamed | Lower peak |
| Relationship types | 3 | 10+ | 3x+ |
| Crash recovery | None | Checkpoint | Full resume |

## Troubleshooting

### Low GPU Utilization
- Increase `--batch-size` (default: 256, try 512 or 1024)
- Ensure CUDA is available: `torch.cuda.is_available()`

### Memory Issues
- Reduce `--preprocess-queue-size` (default: 2048)
- Reduce `--batch-size` for smaller GPU memory

### Network Bottleneck
- Increase `--download-workers` for faster downloads
- Use local file with `--wiki-dump` for best performance

### Neo4j Connection Issues
- Check connection pool: increase `--upload-workers`
- Verify credentials and database name

## Legacy Compatibility

The original `api/scripts/ingest_wiki_dual.py` has been replaced with a thin wrapper that redirects to this module:

```bash
# Both commands are equivalent:
python -m ingestion [args...]
python api/scripts/ingest_wiki_dual.py [args...]
```
