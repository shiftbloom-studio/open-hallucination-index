#!/usr/bin/env node
/* eslint-disable @typescript-eslint/no-require-imports */
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const publicDir = path.join(process.cwd(), 'public');
const input = path.join(publicDir, 'open-graph.jpg');

if (!fs.existsSync(input)) {
  console.error('Error: source image not found at', input);
  console.error('Place your primary open-graph.jpg in the project `public/` folder and re-run this script.');
  process.exit(1);
}

const variants = [
  { file: path.join(publicDir, 'open-graph-1200x630.jpg'), width: 1200, height: 630, quality: 90, format: 'jpeg' },
  { file: path.join(publicDir, 'open-graph-800x418.jpg'), width: 800, height: 418, quality: 85, format: 'jpeg' },
  { file: path.join(publicDir, 'open-graph-1200x630.webp'), width: 1200, height: 630, quality: 80, format: 'webp' },
  { file: path.join(publicDir, 'open-graph-800x418.webp'), width: 800, height: 418, quality: 75, format: 'webp' },
];

(async () => {
  try {
    for (const v of variants) {
      const pipeline = sharp(input).resize(v.width, v.height, { fit: 'cover', position: 'centre' });
      if (v.format === 'webp') {
        await pipeline.webp({ quality: v.quality }).toFile(v.file);
      } else {
        await pipeline.jpeg({ quality: v.quality }).toFile(v.file);
      }
      console.log('Wrote', v.file);
    }
    // Ensure the canonical `open-graph.jpg` is a high-quality 1200x630 JPEG
    const primary = path.join(publicDir, 'open-graph-1200x630.jpg');
    const canonical = path.join(publicDir, 'open-graph.jpg');
    if (fs.existsSync(primary)) {
      fs.copyFileSync(primary, canonical);
      console.log('Overwrote canonical image at', canonical);
    }

    console.log('All variants generated.');
  } catch (err) {
    console.error('Image generation failed:', err);
    process.exit(1);
  }
})();
