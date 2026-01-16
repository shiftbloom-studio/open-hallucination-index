# Frontend API Update Summary

## Changes Made

### 1. API Types Updated ([src/frontend/src/lib/api.ts](src/frontend/src/lib/api.ts))

#### Added Missing Request Fields
Updated `VerifyTextRequest` interface to include:
- ✅ `tier?: "local" | "default" | "max" | null` - Evidence collection tier
- ✅ `skip_decomposition?: boolean` - Skip LLM claim decomposition
- ✅ `return_evidence?: boolean` - Include detailed evidence traces

#### Added New Types
- **`EvidenceSource`** - Enum type for evidence sources (graph, vector, MCP sources)
- **`Evidence`** - Interface for evidence items with:
  - `id`, `source`, `source_id`
  - `content`, `structured_data`
  - `similarity_score`, `match_type`, `classification_confidence`
  - `retrieved_at`, `source_uri`
- **`CitationTrace`** - Interface for provenance trails with:
  - `claim_id`, `status`, `reasoning`
  - `supporting_evidence`, `refuting_evidence`
  - `confidence`, `verification_strategy`

#### Updated Existing Types
- **`VerificationStatus`** - Extended with `"supported" | "partially_supported" | "uncertain"`
- **`ClaimSummary`** - Added `trace?: CitationTrace | null` field
- **`TrustScore`** - Now properly structured with `overall`, `claims_total`, etc.

---

### 2. Citation Trace Viewer Component ([src/frontend/src/components/dashboard/citation-trace-viewer.tsx](src/frontend/src/components/dashboard/citation-trace-viewer.tsx))

Created a beautiful, collapsible visualization component featuring:

#### Features
- **Collapsible Evidence Display** - Show/hide evidence with smooth transitions
- **Source Icons** - Visual indicators for different evidence sources:
  - 🌐 Wikipedia/Wikidata
  - 🗄️ Knowledge Graph
  - 🧪 Academic/PubMed
  - 📰 News sources
  - 📚 Documentation
  - 🛡️ Security databases
- **Evidence Cards** - Separate sections for:
  - ✅ Supporting Evidence (green theme)
  - ❌ Refuting Evidence (red theme)
- **Rich Metadata Display**:
  - Match similarity scores
  - Classification confidence
  - Source URIs with external links
  - Expandable content for long texts
- **Reasoning Analysis** - Highlighted analysis section explaining the verification decision

#### Design
- Consistent with existing dashboard design
- Uses shadcn/ui components (Card, Badge, Button)
- Responsive layout with proper spacing
- Color-coded evidence types
- Smooth animations

---

### 3. Dashboard Form Updates ([src/frontend/src/components/dashboard/verify-ai-output-form.tsx](src/frontend/src/components/dashboard/verify-ai-output-form.tsx))

#### Added Return Evidence Toggle
- New checkbox: "Include detailed evidence and citation traces"
- State: `returnEvidence` (default: `true`)
- Passed to API: `return_evidence` parameter

#### Integrated Citation Trace Viewer
- Claims now display with nested evidence traces
- Automatic rendering when `claim.trace` is present
- Maintains clean visual hierarchy

#### Updated API Call
```typescript
const result = await client.verifyText({
  text,
  context: context || undefined,
  strategy: "adaptive",
  target_sources: targetSources,
  return_evidence: returnEvidence, // ← NEW
});
```

---

### 4. UI Component Added ([src/frontend/src/components/ui/badge.tsx](src/frontend/src/components/ui/badge.tsx))

Created Badge component using `class-variance-authority` with variants:
- `default` - Primary badge style
- `secondary` - Secondary badge style
- `destructive` - Error/warning badge style
- `outline` - Outlined badge style

---

### 5. Test Updates

Updated all test mocks to match new API structure:

#### Files Updated
- ✅ [src/frontend/src/test/mocks/handlers.ts](src/frontend/src/test/mocks/handlers.ts)
- ✅ [src/frontend/src/lib/__tests__/api.test.ts](src/frontend/src/lib/__tests__/api.test.ts)
- ✅ [src/frontend/src/components/dashboard/__tests__/verify-ai-output-form.test.tsx](src/frontend/src/components/dashboard/__tests__/verify-ai-output-form.test.tsx)
- ✅ [src/frontend/src/components/dashboard/__tests__/add-hallucination-form.test.tsx](src/frontend/src/components/dashboard/__tests__/add-hallucination-form.test.tsx)

#### Changes
- Updated `TrustScore` structure (added `overall`, `claims_total`, etc.)
- Added `trace: null` to all mock claims
- Updated test assertions from `trust_score.score` to `trust_score.overall`

---

## Feature Parity Status

### ✅ Fully Implemented
- `text` - Input text
- `context` - Optional context
- `strategy` - Verification strategy
- `tier` - Evidence collection tier
- `use_cache` - Cache usage
- `target_sources` - Source count target
- `skip_decomposition` - Skip claim decomposition
- `return_evidence` - Include evidence traces *(with UI toggle)*
- `CitationTrace` - Evidence visualization *(with beautiful UI)*

### 🎨 User Interface
- Dashboard form includes `return_evidence` toggle only
- `tier` and `skip_decomposition` are available in API but not exposed in UI (as requested)
- Citation traces render automatically when present
- Evidence cards are expandable, color-coded, and include source icons

---

## Benefits

1. **Full API Compatibility** - Frontend now supports all backend API fields
2. **Enhanced Transparency** - Users can see evidence sources and reasoning
3. **Better UX** - Toggle evidence on/off to reduce response size when not needed
4. **Visual Appeal** - Beautiful, professional evidence visualization
5. **Type Safety** - Complete TypeScript types for all new structures
6. **Test Coverage** - All tests updated and passing

---

## Usage Example

```tsx
// Request with evidence
const result = await client.verifyText({
  text: "The Eiffel Tower is in Paris",
  return_evidence: true,
  target_sources: 6,
});

// Access evidence
result.claims.forEach(claim => {
  if (claim.trace) {
    console.log(`Supporting: ${claim.trace.supporting_evidence.length}`);
    console.log(`Refuting: ${claim.trace.refuting_evidence.length}`);
  }
});
```

---

## Next Steps (Optional)

1. Add UI controls for `tier` parameter (local/default/max)
2. Add UI toggle for `skip_decomposition`
3. Add evidence filtering (by source type)
4. Add evidence export functionality
5. Create 3D knowledge graph visualization (using existing knowledge-track API)

---

## Files Changed

1. ✅ `src/frontend/src/lib/api.ts`
2. ✅ `src/frontend/src/components/dashboard/citation-trace-viewer.tsx` (NEW)
3. ✅ `src/frontend/src/components/ui/badge.tsx` (NEW)
4. ✅ `src/frontend/src/components/dashboard/verify-ai-output-form.tsx`
5. ✅ `src/frontend/src/test/mocks/handlers.ts`
6. ✅ `src/frontend/src/lib/__tests__/api.test.ts`
7. ✅ `src/frontend/src/components/dashboard/__tests__/verify-ai-output-form.test.tsx`
8. ✅ `src/frontend/src/components/dashboard/__tests__/add-hallucination-form.test.tsx`

Total: **6 files updated, 2 files created**
