# Wikipedia Ingestion Pipeline

High-performance Wikipedia ingestion pipeline for the Open Hallucination Index project.

## Features

- **10-50x faster** than the original monolithic script
- **Parallel dump processing** with configurable worker count
- **Producer-consumer architecture** with non-blocking queues
- **Parallel downloads** with resume support for Wikipedia dumps
- **GPU-accelerated embeddings** with dedicated embedding workers
- **Async uploads** to both Qdrant and Neo4j
- **10+ relationship types** in Neo4j knowledge graph
- **Resumable checkpoints** for crash recovery
- **Real-time progress** with rich statistics

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  Download   │───▶│ Dump Workers │───▶│   Embed     │───▶│   Upload    │
│  (4 threads)│    │ (2 workers)  │    │ (2 workers) │    │ (4 threads) │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
      │                  │                   │                   │
      ▼                  ▼                   ▼                   ▼
   Wikipedia         Parallel Parse      GPU Encode        Qdrant + Neo4j
    Dumps            + Preprocess        + BM25 Sparse     Parallel Upload
```

### Multi-Worker Architecture

The pipeline now supports **parallel dump file processing**:

- **Dump Workers** (`--dump-workers`): Multiple dump files are processed simultaneously
- **Embedding Workers** (`--embedding-workers`): Dedicated GPU threads for embedding computation
- **Preprocess Workers** (`--preprocess-workers`): Parallel text preprocessing
- **Upload Workers** (`--upload-workers`): Async database uploads

This eliminates freezing during batch processing by decoupling all stages.

## Quick Start

```bash
# Run from project root
cd src/ingestion
python -m ingestion --help

# Basic usage with defaults
python -m ingestion --limit 10000

# Balanced settings (recommended for 64GB RAM + RTX 4090)
python -m ingestion \
    --batch-size 384 \
    --dump-workers 3 \
    --preprocess-workers 12 \
    --embedding-workers 3 \
    --upload-workers 2 \
    --embedding-batch-size 768 \
    --embedding-device cuda

# Endless mode with auto-retry on network errors
python -m ingestion --endless --keep-downloads
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
    batch_size=256,              # Articles per batch
    chunk_size=512,              # Characters per chunk
    chunk_overlap=64,            # Overlap between chunks
    
    # Worker configuration
    dump_workers=2,              # Parallel dump file workers
    download_workers=4,          # Parallel download threads
    preprocess_workers=8,        # Text processing threads
    embedding_workers=2,         # GPU embedding workers
    upload_workers=4,            # Upload threads per store
    
    # Queue sizes
    download_queue_size=8,       # Pending downloads
    preprocess_queue_size=2048,  # Pending articles
    upload_queue_size=16,        # Pending batches
    
    # Embedding settings
    embedding_batch_size=512,    # GPU batch size
    embedding_device="cuda",     # "cuda", "cpu", or "auto"
)
```

### Recommended Settings by Hardware

| RAM | GPU | dump_workers | embedding_workers | batch_size |
|-----|-----|--------------|-------------------|------------|
| 16GB | None | 1 | 1 | 128 |
| 32GB | 8GB | 2 | 2 | 256 |
| 64GB | 12GB+ | 3 | 3 | 384 |
| 128GB | 24GB+ | 8 | 8 | 1024 |

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
python src/api/scripts/ingest_wiki_dual.py [args...]
```
