# Citation Trace Viewer - Visual Guide

## Overview

The Citation Trace Viewer provides a beautiful, interactive visualization of evidence sources used to verify claims. It displays supporting and refuting evidence with rich metadata and interactive features.

## Features

### 🎨 Visual Design

#### Color-Coded Evidence
- **Supporting Evidence**: Green theme (`bg-green-500/5`, `border-green-500/20`)
- **Refuting Evidence**: Red theme (`bg-red-500/5`, `border-red-500/20`)

#### Source Icons
Each evidence source displays a contextual icon:
- 🌐 **Globe** - Wikipedia, Wikidata, MediaWiki
- 🗄️ **Database** - Knowledge Graph, Neo4j
- 🧪 **Flask** - PubMed, NCBI, Academic sources
- 📰 **Newspaper** - News sources, GDELT
- 📚 **Book** - Documentation, OpenAlex, Crossref
- 🛡️ **Shield** - Security databases, OSV
- 📊 **Activity** - Other sources

### 📊 Metadata Display

Each evidence card shows:
1. **Source Badge** - Formatted source name with icon
2. **Match Score** - Similarity percentage (when available)
3. **Confidence Score** - Classification confidence (when available)
4. **Content** - Evidence text with expandable preview
5. **Source URI** - External link to original source

### 🎯 Interactive Elements

#### Collapsible Container
- Click "Hide" to collapse the entire evidence section
- Click "Show" to expand
- Smooth transitions

#### Expandable Content
- Long evidence texts are truncated to 2 lines
- "Show more" / "Show less" buttons for full content
- Character threshold: 150 characters

### 📐 Layout Structure

```
┌─ Citation Trace Card ─────────────────────────────────┐
│ Evidence Trail                           [Hide/Show]  │
│ 2 sources analyzed • Strategy: adaptive               │
├───────────────────────────────────────────────────────┤
│                                                        │
│ ┌─ Analysis Section ──────────────────────────────┐  │
│ │ This claim is supported by evidence from        │  │
│ │ multiple reliable sources including...          │  │
│ └─────────────────────────────────────────────────┘  │
│                                                        │
│ ✅ Supporting Evidence (2)                            │
│ ┌─────────────────────────────────────────────────┐  │
│ │ ✓ [🌐 Wikipedia] Match: 95% Confidence: 90%    │  │
│ │   Wikipedia confirms this fact with detailed... │  │
│ │   [Show more] [View source ↗]                   │  │
│ └─────────────────────────────────────────────────┘  │
│ ┌─────────────────────────────────────────────────┐  │
│ │ ✓ [🗄️ Graph Exact] Match: 100%                 │  │
│ │   Knowledge graph has exact match for entity... │  │
│ └─────────────────────────────────────────────────┘  │
│                                                        │
│ ❌ Refuting Evidence (0)                              │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Usage Example

### Basic Usage

```tsx
import CitationTraceViewer from '@/components/dashboard/citation-trace-viewer';

function MyComponent({ claim }) {
  if (!claim.trace) return null;
  
  return (
    <CitationTraceViewer 
      trace={claim.trace} 
      claimText={claim.text} 
    />
  );
}
```

### With API Response

```tsx
const result = await client.verifyText({
  text: "The Eiffel Tower is in Paris",
  return_evidence: true,
});

result.claims.forEach(claim => {
  if (claim.trace) {
    // Render CitationTraceViewer
    <CitationTraceViewer 
      trace={claim.trace} 
      claimText={claim.text} 
    />
  }
});
```

## Data Structure

### CitationTrace Interface
```typescript
interface CitationTrace {
  claim_id: string;
  status: VerificationStatus;
  reasoning: string;
  supporting_evidence: Evidence[];
  refuting_evidence: Evidence[];
  confidence: number;
  verification_strategy: string;
}
```

### Evidence Interface
```typescript
interface Evidence {
  id: string;
  source: EvidenceSource;
  source_id?: string | null;
  content: string;
  structured_data?: Record<string, unknown> | null;
  similarity_score?: number | null;
  match_type?: string | null;
  classification_confidence?: number | null;
  retrieved_at: string;
  source_uri?: string | null;
}
```

## Component Props

```typescript
interface CitationTraceViewerProps {
  trace: CitationTrace;
  claimText: string;
}
```

## Styling

### Color Palette
- **Background**: `bg-slate-800/30` with `border-slate-700/50`
- **Success**: `text-green-500`, `bg-green-500/10`
- **Error**: `text-red-500`, `bg-red-500/10`
- **Info**: `bg-slate-700/30`, `border-slate-600/30`

### Typography
- **Title**: 16px (text-base), medium weight
- **Description**: 12px (text-xs), muted
- **Content**: 14px (text-sm)
- **Badges**: 12px (text-xs), semibold

### Spacing
- **Card Padding**: 12px (p-3) / 16px (p-4)
- **Gap**: 8px (gap-2) / 16px (gap-4)
- **Rounded**: 8px (rounded-lg)

## Accessibility

- ✅ Semantic HTML structure
- ✅ ARIA-compliant buttons
- ✅ Keyboard navigation support
- ✅ Screen reader friendly
- ✅ Color contrast meets WCAG AA standards
- ✅ External links open in new tab with `rel="noopener noreferrer"`

## Performance

- **Lazy Rendering**: Evidence cards only render when visible
- **Optimized Re-renders**: Uses React state for collapsible sections
- **Efficient Layout**: CSS Grid and Flexbox for responsive design
- **No Heavy Dependencies**: Uses native browser features

## Browser Support

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Testing

Comprehensive test coverage includes:
- ✅ Rendering with supporting evidence
- ✅ Rendering with refuting evidence
- ✅ Collapsible functionality
- ✅ External link generation
- ✅ Score display
- ✅ Empty state handling

See: `src/frontend/src/components/dashboard/__tests__/citation-trace-viewer.test.tsx`

## Integration with Dashboard

The component automatically integrates into the verify-ai-output-form:

```tsx
{verificationResult.claims.map((claim) => (
  <div key={claim.id} className="space-y-2">
    <div className="p-4 rounded-lg border">
      {/* Claim display */}
    </div>
    {claim.trace && (
      <CitationTraceViewer 
        trace={claim.trace} 
        claimText={claim.text} 
      />
    )}
  </div>
))}
```

## Future Enhancements

Potential improvements:
- 📊 Evidence source statistics
- 🔍 Filter by source type
- 📤 Export evidence to JSON/CSV
- 🔗 Link to knowledge graph visualization
- ⭐ Evidence quality indicators
- 🏷️ Tag-based evidence grouping
