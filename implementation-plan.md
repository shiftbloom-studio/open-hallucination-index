## 1. Project Structure & Assets
Create src/components/landing directory to house specific landing page components.
Ensure lenis is configured for smooth scrolling in the main layout or just this page if possible.

## 2. Components (src/components/landing/)
[NEW] 
HeroSection.tsx
Visuals: A 3D particle network background using @react-three/fiber and @react-three/drei. Represents the "Knowledge Graph".
Content: High-impact headline: "The Trust Layer for Artificial Intelligence."
Interaction: Mouse movement affects the particle field.
[NEW] 
ProblemSection.tsx
Visuals: Text-based scroll reveal using framer-motion. High contrast.
Content: "LLMs Hallucinate. We Verify." - Explaining the problem of "confidently wrong" AI.
[NEW] 
ArchitectureFlow.tsx
Visuals: A step-by-step animated diagram showing the flow:
Input Text
Claim Decomposition (Atomization)
Verification Oracle (Graph + Vector Search)
Trust Score
Tech: SVG animations with Framer Motion.
[NEW] 
FeatureGrid.tsx
Visuals: Bento-grid style layout with glassmorphism cards.
Content:
GraphRAG: "Beyond simple vector search."
Atomic Verification: " Checking facts, not just tokens."
Open Source: "Transparent and community-driven."
[NEW] 
CtaSection.tsx
Visuals: Clean, bold typography with a glowing "Get Started" or "View on GitHub" button.
Background: Subtle gradient pulse.
## 3. Page Assembly (src/app/page.tsx)
Refactor existing page.tsx to compose these new sections.
Wrap in a SmoothScroll provider (Lenis).
Verification Plan
Automated Tests
Run npm run build to ensure no type errors or build failures with the new 3D components.