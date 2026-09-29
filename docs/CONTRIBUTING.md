# Contributing to Open Hallucination Index

First off, thank you for considering contributing to Open Hallucination Index! 🎉

## First evening

You do not need to be good. Finish the small thing.
English or German is fine.
Docs is a valid first door.

Pick a [good first issue](https://github.com/shiftbloom-studio/open-hallucination-index/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22), comment on it, and open a focused PR.

If the rest of this guide feels heavy, start at https://github.com/shiftbloom-studio/open-hallucination-index/issues/52

## First Worker evening

Production is the Cloudflare Worker in `cloudflare/ohi-worker/`. The FastAPI app in `src/api/` (`ohi-server`) is a legacy local stack. You do not need Docker, Neo4j, Qdrant, Redis, or `ohi-server` to work on the Worker.

| | Production Worker | `ohi-server` (`src/api`) |
|---|---|---|
| Runtime | Cloudflare Workers (`wrangler`) | FastAPI / Python |
| Public API | `/api/v2` | `/api/v1` |
| Stores | D1, Vectorize, R2, Durable Objects | Neo4j, Qdrant, Redis |
| In the live request path? | Yes (`https://ohi.shiftbloom.studio`) | No |

Topology and the short recommended loop live in [CURRENT_ARCHITECTURE.md](CURRENT_ARCHITECTURE.md) §4.

### Install and the safe wrangler commands

This package has its own lockfile. Use `pnpm` from `cloudflare/ohi-worker/` (CI uses Node 24).

```bash
cd cloudflare/ohi-worker
pnpm install
pnpm run types   # wrangler types -> src/worker-configuration.d.ts
pnpm run check   # tsc --noEmit
pnpm run build   # wrangler deploy --dry-run --outdir dist
```

`pnpm run build` compiles the Worker without publishing it. That is the first-evening check.

`pnpm run dev` runs `wrangler dev` (local Miniflare by default). Pair it with the frontend loop from CURRENT_ARCHITECTURE §4 (`NEXT_PUBLIC_API_BASE=http://localhost:8787/api/v2`). `wrangler.jsonc` serves static assets from `src/frontend/out`; build that export if you need the Worker to serve pages.

Do not run these on a first evening (they touch production):

- `pnpm run deploy` / `wrangler deploy`
- `wrangler d1 migrations apply ohi-prod --remote`
- `wrangler secret put`
- `wrangler dev --remote` (uses deployed Cloudflare resources / the production zone edge)

### Bindings and secrets without production credentials

`wrangler.jsonc` does not set `remote: true` on bindings. Local `wrangler dev` therefore uses local simulations for D1 (`OHI_DB`), R2 (`OHI_CORPUS`), Queues, Durable Objects, and Workflows — empty, not a copy of prod. Vectorize (`OHI_VECTOR`) has no local simulation. Workers AI (`AI`) is not a local model; calls go to Cloudflare when you are authenticated.

`ADMIN_TOKEN` and `TURNSTILE_SECRET_KEY` are Worker secrets, not `wrangler.jsonc` vars. Do not put production values in the repo.

Without those secrets, the code fails closed or skips as follows (`cloudflare/ohi-worker/src/index.ts`):

- `ADMIN_TOKEN` missing: `GET/POST /api/v2/admin/*` returns `503` with `admin_token_not_configured`
- `TURNSTILE_SECRET_KEY` missing: `rejectIfAbusive` skips the Turnstile check and does not challenge the request
- `GET /health/live`: no bindings required
- `GET /health/ready`: probes local D1 and Durable Objects; it marks Workers AI and Vectorize as `configured` without calling them
- `GET /health/deep`: actually calls Workers AI, Wikipedia, and Vectorize — expect `degraded` locally when Vectorize/AI are not remote

Do not publish secrets. Do not try to bypass the production WAF or Turnstile.

## 📋 Table of Contents

- [First evening](#first-evening)
- [First Worker evening](#first-worker-evening)
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to creating a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/shiftbloom-studio/open-hallucination-index.git
   cd open-hallucination-index
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/shiftbloom-studio/open-hallucination-index.git
   ```

## Development Setup

### Prerequisites

- Python 3.14+ (for the API)
- Node.js 22+ (for the frontend)
- Git

### API Setup

```bash
cd src/api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Frontend Setup

```bash
cd src/frontend

# Install dependencies
npm install
```

### Running Tests

**API:**
```bash
cd src/api

# Run all tests
pytest

# Run with coverage
pytest --cov=src/open_hallucination_index --cov-report=term-missing

# Run specific test file
pytest tests/integration/test_api.py -v
```

**Core/Repo Tests (root):**
```bash
# From repository root
pytest src/api/tests/ -v
```

**Frontend:**
```bash
cd src/frontend

# Run unit tests
npm run test

# Run E2E tests
npm run test:e2e
```

### Code Quality Tools

**API (Python):**
```bash
cd src/api

# Linting
ruff check src tests

# Fix auto-fixable issues
ruff check src tests --fix

# Formatting
ruff format src tests

# Type checking
mypy src
```

**Frontend (TypeScript):**
```bash
cd src/frontend

# Linting
npm run lint

# Type checking
npx tsc --noEmit
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-verification-strategy`
- `fix/correct-score-calculation`
- `docs/update-api-documentation`
- `refactor/improve-cache-performance`

### Commit Messages

Follow conventional commits format:

```
type(scope): short description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(scorer): add evidence-ratio based scoring`
- `fix(mcp): handle connection timeout gracefully`
- `docs(readme): update quick start guide`

## Pull Request Process

1. **Update your fork** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** and commit them

4. **Run tests and linting**:
   ```bash
   pytest
   ruff check src tests
   mypy src
   ```

5. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub with:
   - Clear title and description
   - Reference to any related issues
   - Screenshots/examples if applicable

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows project style guidelines
- [ ] Documentation is updated (if applicable)
- [ ] Commit messages are clear and follow conventions
- [ ] No unnecessary files are included

## Style Guidelines

### Python Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Key points:

- **Line length**: 100 characters
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with isort rules
- **Docstrings**: Google style

Example:
```python
async def verify_claim(
    self,
    claim: Claim,
    strategy: VerificationStrategy | None = None,
) -> tuple[VerificationStatus, CitationTrace]:
    """
    Verify a single claim against knowledge sources.

    Args:
        claim: The claim to verify.
        strategy: Verification strategy to use (or default).

    Returns:
        Tuple of (verification status, citation trace with evidence).
    """
    ...
```

### Type Hints

- Use type hints for all function signatures
- Use `from __future__ import annotations` for forward references
- Prefer `X | None` over `Optional[X]`

### Architecture

Follow the hexagonal architecture pattern:
- **Domain**: Pure business logic, no external dependencies
- **Ports**: Abstract interfaces
- **Adapters**: Concrete implementations
- **Application**: Use-case orchestration

## Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear explanation of the bug
2. **Steps to Reproduce**: Minimal steps to reproduce
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, Docker version
6. **Logs**: Relevant error messages or logs

Use the bug report template when creating an issue.

## Suggesting Features

Feature requests are welcome! Please include:

1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: How would you solve it?
3. **Alternatives Considered**: Other approaches you've thought about
4. **Additional Context**: Any other relevant information

---

Thank you for contributing! 🚀
