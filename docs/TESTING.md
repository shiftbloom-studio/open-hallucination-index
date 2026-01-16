# Testing Infrastructure - Open Hallucination Index

## Overview

Comprehensive test suite for all Python subprojects with CI/CD integration.

## Test Structure

### API (`src/api/tests/`)

```
tests/
├── unit/
│   ├── adapters/           # Adapter layer tests (NEW)
│   │   ├── test_neo4j_adapter.py
│   │   ├── test_qdrant_adapter.py
│   │   ├── test_redis_adapter.py
│   │   └── test_llm_adapter.py
│   ├── domain/             # Domain entity tests
│   │   ├── test_entities.py
│   │   ├── test_results.py
│   │   └── test_scorers.py
│   └── application/        # Use case tests
│       └── test_claim_decomposition.py
├── integration/            # API integration tests
│   └── test_api.py
└── conftest.py            # Shared fixtures
```

**Coverage Target**: 70% (configured in `pyproject.toml`)

### Ingestion (`src/ingestion/tests/`) - NEW

```
tests/
├── test_downloader.py      # Wikipedia downloader tests
├── test_preprocessor.py    # Text preprocessing tests
├── test_models.py          # Data model tests
└── conftest.py            # Fixtures (wiki articles, configs)
```

**Coverage Target**: 60% (configured in `pyproject.toml`)

### Benchmark (`src/benchmark/tests/`) - NEW

```
tests/
├── test_evaluators.py      # Evaluator tests
├── test_config.py          # Configuration tests
└── conftest.py            # Fixtures (responses, datasets)
```

**Coverage Target**: 60% (configured in `pyproject.toml`)

## Running Tests

### Local Execution

#### API Tests
```bash
cd src/api
source .venv/bin/activate  # or .venv/Scripts/activate on Windows
pytest                     # All tests
pytest tests/unit/         # Unit tests only
pytest --cov               # With coverage
```

#### Ingestion Tests
```bash
cd src/ingestion
source .venv/bin/activate
pytest
```

#### Benchmark Tests
```bash
cd src/benchmark
pytest
```

### Coverage Reports

All subprojects configured with coverage thresholds:

```bash
# Generate HTML coverage report
pytest --cov --cov-report=html

# View report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html # Windows
```

### Pre-commit Hooks

Install once:
```bash
pip install pre-commit
pre-commit install
```

Run manually:
```bash
pre-commit run --all-files
```

Hooks include:
- **Ruff**: Linting + formatting
- **MyPy**: Type checking
- **Prettier**: Frontend formatting
- **ESLint**: Frontend linting
- **detect-secrets**: Secret scanning
- **markdownlint**: Markdown formatting

## GitHub Actions CI/CD

### Workflows

1. **API CI** (`.github/workflows/ci.yml`)
   - Triggers: `src/api/**` changes
   - Jobs: lint, test (70% coverage), security
   - Codecov upload: ✅

2. **Ingestion CI** (`.github/workflows/ingestion-ci.yml`) - NEW
   - Triggers: `src/ingestion/**` changes
   - Jobs: lint, test (60% coverage), security
   - Codecov upload: ✅

3. **Benchmark CI** (`.github/workflows/benchmark-ci.yml`) - NEW
   - Triggers: `src/benchmark/**` changes
   - Jobs: lint, test (60% coverage), security
   - Codecov upload: ✅

4. **Frontend CI** (`.github/workflows/frontend.yml`)
   - Triggers: `src/frontend/**` changes
   - Jobs: lint, test (75% coverage), build, e2e
   - Coverage: Vitest + Playwright

### CI Strategy

- **Fast feedback**: Lint runs first, blocks on failure
- **Parallel jobs**: Test and security run concurrently
- **Mock dependencies**: No external services (Neo4j, Qdrant, Redis)
- **Trimmed execution**: Unit tests only, no integration tests in CI

## Test Principles

### 1. Unit Tests (Fast, Isolated)

- **Mock external dependencies** (databases, APIs, LLMs)
- **Focus on business logic**
- **No network calls**
- **Execution time**: < 5 seconds per test

### 2. Fixtures (Reusable Test Data)

Each subproject has `conftest.py` with shared fixtures:

**API fixtures**:
- `sample_claim`, `sample_evidence`, `sample_verification_result`
- `mock_llm_client`, `mock_neo4j_driver`, `mock_qdrant_client`

**Ingestion fixtures**:
- `sample_wiki_article`, `sample_wiki_section`, `sample_wiki_infobox`
- `mock_neo4j_driver`, `mock_qdrant_client`, `mock_sentence_transformer`

**Benchmark fixtures**:
- `sample_verification_response`, `sample_benchmark_dataset`
- `mock_httpx_client`

### 3. Adapter Testing Strategy

**New adapter tests** follow hexagonal architecture:

```python
# ✅ Good: Mock at adapter boundary
@pytest.fixture
def neo4j_store(mock_neo4j_driver):
    store = Neo4jGraphStore(...)
    store.driver = mock_neo4j_driver  # Inject mock
    return store

# ✅ Test adapter behavior, not driver
async def test_find_evidence(neo4j_store):
    evidence = await neo4j_store.find_evidence("claim")
    assert isinstance(evidence[0], Evidence)
```

### 4. Coverage Thresholds

| Subproject | Lines | Statements | Functions | Branches |
|------------|-------|------------|-----------|----------|
| API        | 70%   | 70%        | -         | -        |
| Ingestion  | 60%   | 60%        | -         | -        |
| Benchmark  | 60%   | 60%        | -         | -        |
| Frontend   | 75%   | 75%        | 70%       | 60%      |

Configured in each `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = """
    --cov=<package>
    --cov-fail-under=<threshold>
    --cov-report=xml
    --cov-report=term-missing
"""
```

## Dependencies

### API
- `pytest>=9.0.2`
- `pytest-asyncio>=1.3.0`
- `pytest-cov>=7.0.0`

### Ingestion
- `pytest>=7.0.0`
- `pytest-asyncio>=0.21.0`
- `pytest-cov>=7.0.0`

### Benchmark
- `pytest>=7.0.0`
- `pytest-asyncio>=0.21.0`
- (no coverage dependency yet)

Install dev dependencies:
```bash
pip install -e ".[dev]"
```

## Best Practices

### ✅ DO

- Write unit tests for all new features
- Mock external dependencies (DB, API, LLM)
- Use async tests for async code (`@pytest.mark.asyncio`)
- Parametrize tests for multiple scenarios
- Keep tests fast (< 5s each)
- Use descriptive test names (`test_<what>_<scenario>`)

### ❌ DON'T

- Test external libraries (trust Neo4j, Qdrant work)
- Write integration tests requiring real services (for now)
- Use `time.sleep()` in tests
- Test implementation details (private methods)
- Ignore coverage thresholds

## Troubleshooting

### Import Errors in Tests

**Problem**: `Import "pytest" could not be resolved`

**Solution**: Install dev dependencies in venv
```bash
pip install -e ".[dev]"
```

### Coverage Fails Below Threshold

**Problem**: `FAIL Required test coverage of X% not reached`

**Solution**: Add tests or adjust threshold in `pyproject.toml`

### Pre-commit Fails on MyPy

**Problem**: Type errors block commit

**Solution**: Fix types or add `# type: ignore` (temporary)

### GitHub Actions Fails

**Problem**: Tests pass locally, fail in CI

**Solution**: Check CI environment variables, dependencies

## Future Enhancements

- [ ] Integration tests with Docker Compose
- [ ] Contract tests (API ↔ Frontend)
- [ ] Performance tests (load testing)
- [ ] Mutation testing
- [ ] Test data factories (Faker, Factory Boy)
- [ ] Snapshot testing for API responses

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [pre-commit hooks](https://pre-commit.com/)
- [Hexagonal Architecture Testing](https://alistair.cockburn.us/hexagonal-architecture/)
