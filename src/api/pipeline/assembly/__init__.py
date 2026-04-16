"""L7 output assembly. Spec §9.

Maps L5 CalibratedVerdicts + L4 PCG metadata into the public
``ClaimVerdict`` / ``DocumentVerdict`` schema. The Gaussian copula
aggregator is the only non-trivial piece here; everything else is
pure plumbing.
"""

from __future__ import annotations

from pipeline.assembly.claim_verdict import assemble_claim_verdict
from pipeline.assembly.copula import gaussian_copula_joint, nearest_psd
from pipeline.assembly.document_verdict import assemble_document_verdict

__all__ = [
    "assemble_claim_verdict",
    "assemble_document_verdict",
    "gaussian_copula_joint",
    "nearest_psd",
]
