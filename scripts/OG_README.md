OG image generation
===================

This project includes a small Node script to generate optimized Open Graph images from a source `public/open-graph.jpg`.

Usage:

1. Ensure the source image `public/open-graph.jpg` exists (preferably a high-resolution master image).
2. Run:

```bash
npm run og:generate
```

What it does:
- Produces 1200x630 and 800x418 variants in both JPEG and WebP.
- Overwrites `public/open-graph.jpg` with a high-quality 1200x630 JPEG so the canonical image is consistent.

Tips:
- Keep an editable master image outside the repo if you want to re-design the OG image.
- WebP is preferred where supported; the metadata includes WebP variants first for better performance.
