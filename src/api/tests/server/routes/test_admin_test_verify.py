import pytest

from server.routes import admin


async def test_test_verify_uses_computed_score_for_supported_claim() -> None:
    response = await admin.test_verify(admin.TestVerifyRequest(test_type="simple"), True)

    assert response.success is True
    assert response.claims_found == 1
    assert response.verification_score == pytest.approx(0.991, abs=0.0001)


async def test_test_verify_uses_computed_score_for_refuted_claims() -> None:
    response = await admin.test_verify(admin.TestVerifyRequest(test_type="hallucination"), True)

    assert response.success is True
    assert response.claims_found == 3
    assert response.verification_score == pytest.approx(0.0, abs=0.0001)
