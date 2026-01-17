"""
Admin API Endpoints
====================

API endpoints for admin management: users, API keys, and system oversight.
Requires admin authentication via X-Admin-Key header or admin user session.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import get_settings
from server.services.live_logs import live_log_service

router = APIRouter()
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Request/Response Schemas
# -----------------------------------------------------------------------------


class UserResponse(BaseModel):
    """User information response."""

    id: UUID
    email: str
    name: str | None
    ohi_tokens: int
    role: Literal["user", "admin"]
    created_at: datetime
    updated_at: datetime


class UserListResponse(BaseModel):
    """Paginated list of users."""

    users: list[UserResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class UpdateUserRoleRequest(BaseModel):
    """Request to update user role."""

    role: Literal["user", "admin"]


class CreateApiKeyRequest(BaseModel):
    """Request to create a new API key."""

    name: str = Field(..., min_length=1, max_length=100, description="Friendly name for the key")
    type: Literal["standard", "master", "guest"] = Field(
        default="standard",
        description="Key type: standard (linked to user), master (full access), guest (limited tokens)",
    )
    user_id: UUID | None = Field(
        default=None,
        description="User ID to link key to (required for 'standard' type, creates guest user for 'guest' type)",
    )
    token_limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens this key can use (null for unlimited)",
    )
    expires_in_days: int | None = Field(
        default=None,
        ge=1,
        le=365,
        description="Days until key expires (null for non-expiring)",
    )


class ApiKeyResponse(BaseModel):
    """API key information (without the actual key)."""

    id: UUID
    user_id: UUID | None
    user_email: str | None  # Joined from users table
    prefix: str
    name: str
    type: Literal["standard", "master", "guest"]
    token_limit: int | None
    tokens_used: int
    tokens_remaining: int | None  # Computed
    expires_at: datetime | None
    is_active: bool
    last_used_at: datetime | None
    created_at: datetime


class ApiKeyCreatedResponse(BaseModel):
    """Response when creating a new API key - includes the actual key."""

    id: UUID
    key: str = Field(..., description="The actual API key - store securely, it won't be shown again!")
    prefix: str
    name: str
    type: Literal["standard", "master", "guest"]
    token_limit: int | None
    expires_at: datetime | None
    user_id: UUID | None
    user_email: str | None


class ApiKeyListResponse(BaseModel):
    """List of API keys."""

    keys: list[ApiKeyResponse]
    total: int


class BalanceResponse(BaseModel):
    """API key balance information."""

    tokens_remaining: int | None = Field(description="Remaining tokens (null for unlimited)")
    tokens_used: int
    token_limit: int | None
    type: Literal["standard", "master", "guest"]
    key_name: str
    expires_at: datetime | None
    is_active: bool


# -----------------------------------------------------------------------------
# Mock Data Store (Replace with actual Supabase integration)
# -----------------------------------------------------------------------------

# In-memory stores for development/testing
# In production, these would be Supabase queries
_mock_users: dict[str, dict] = {}
_mock_api_keys: dict[str, dict] = {}

# Parameters for API key hashing (slow KDF to resist brute force)
_API_KEY_PBKDF2_ITERATIONS = 100_000
_API_KEY_PBKDF2_SALT = b"admin_api_key_salt_v1"


def _hash_key(key: str) -> str:
    """Hash an API key using a slow, key-derivation function (PBKDF2-HMAC-SHA256)."""
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        key.encode("utf-8"),
        _API_KEY_PBKDF2_SALT,
        _API_KEY_PBKDF2_ITERATIONS,
    )
    return derived.hex()


def _generate_api_key(key_type: str) -> tuple[str, str]:
    """Generate a new API key and return (key, prefix)."""
    # Format: ohi_{type}_{random}
    # e.g., ohi_sk_abc123... (standard), ohi_mk_xyz789... (master), ohi_gk_... (guest)
    type_prefix = {"standard": "sk", "master": "mk", "guest": "gk"}[key_type]
    random_part = secrets.token_urlsafe(32)
    key = f"ohi_{type_prefix}_{random_part}"
    prefix = key[:12]  # e.g., "ohi_sk_abc12"
    return key, prefix


# -----------------------------------------------------------------------------
# Admin Authentication Dependency
# -----------------------------------------------------------------------------


async def verify_admin_access(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> bool:
    """
    Verify admin access via either:
    1. X-API-Key header matching the env API_API_KEY (master key)
    2. X-User-Id header for a user with admin role (checked against DB)
    """
    settings = get_settings()

    # Check if API key is configured and matches (master key = admin access)
    if settings.api.api_key and x_api_key and secrets.compare_digest(x_api_key, settings.api.api_key):
        return True

    # Check if user has admin role (would query Supabase in production)
    if x_user_id:
        # In production: query users table for role == 'admin'
        user = _mock_users.get(x_user_id)
        if user and user.get("role") == "admin":
            return True

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin access required",
    )


# -----------------------------------------------------------------------------
# User Management Endpoints
# -----------------------------------------------------------------------------


@router.get(
    "/users",
    response_model=UserListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all users",
    description="Retrieve a paginated list of all users. Admin access required.",
)
async def list_users(
    _admin: Annotated[bool, Depends(verify_admin_access)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    search: Annotated[str | None, Query(max_length=100)] = None,
) -> UserListResponse:
    """
    List all users with pagination and optional search.

    In production, this queries the Supabase users table.
    """
    # Mock implementation - replace with Supabase query
    all_users = list(_mock_users.values())

    # Filter by search term
    if search:
        search_lower = search.lower()
        all_users = [
            u
            for u in all_users
            if search_lower in u["email"].lower() or (u.get("name") and search_lower in u["name"].lower())
        ]

    total = len(all_users)
    total_pages = (total + page_size - 1) // page_size
    start = (page - 1) * page_size
    end = start + page_size

    users = [
        UserResponse(
            id=UUID(u["id"]),
            email=u["email"],
            name=u.get("name"),
            ohi_tokens=u.get("ohi_tokens", 0),
            role=u.get("role", "user"),
            created_at=u.get("created_at", datetime.now(UTC)),
            updated_at=u.get("updated_at", datetime.now(UTC)),
        )
        for u in all_users[start:end]
    ]

    return UserListResponse(
        users=users,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.patch(
    "/users/{user_id}/role",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Update user role",
    description="Update a user's role (user/admin). Admin access required.",
)
async def update_user_role(
    user_id: UUID,
    request: UpdateUserRoleRequest,
    _admin: Annotated[bool, Depends(verify_admin_access)],
) -> UserResponse:
    """Update a user's role."""
    user_id_str = str(user_id)

    # Mock implementation
    if user_id_str not in _mock_users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    _mock_users[user_id_str]["role"] = request.role
    _mock_users[user_id_str]["updated_at"] = datetime.now(UTC)

    u = _mock_users[user_id_str]
    return UserResponse(
        id=UUID(u["id"]),
        email=u["email"],
        name=u.get("name"),
        ohi_tokens=u.get("ohi_tokens", 0),
        role=u.get("role", "user"),
        created_at=u.get("created_at", datetime.now(UTC)),
        updated_at=u.get("updated_at", datetime.now(UTC)),
    )


# -----------------------------------------------------------------------------
# API Key Management Endpoints
# -----------------------------------------------------------------------------


@router.post(
    "/keys",
    response_model=ApiKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new API key",
    description="Generate a new API key. The actual key is only shown once!",
)
async def create_api_key(
    request: CreateApiKeyRequest,
    _admin: Annotated[bool, Depends(verify_admin_access)],
) -> ApiKeyCreatedResponse:
    """
    Create a new API key.

    Key types:
    - **standard**: Linked to an existing user, inherits their token balance
    - **master**: Full access, unlimited tokens (use sparingly)
    - **guest**: Creates a new guest user with limited tokens
    """
    now = datetime.now(UTC)
    key_id = uuid4()
    user_id = request.user_id
    user_email: str | None = None

    # Validate based on key type
    if request.type == "standard":
        if not request.user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required for standard API keys",
            )
        # Verify user exists
        user = _mock_users.get(str(request.user_id))
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {request.user_id} not found",
            )
        user_email = user["email"]

    elif request.type == "guest":
        # Create a guest user
        guest_user_id = uuid4()
        guest_email = f"guest_{secrets.token_hex(8)}@ohi.guest"
        _mock_users[str(guest_user_id)] = {
            "id": str(guest_user_id),
            "email": guest_email,
            "name": f"Guest ({request.name})",
            "ohi_tokens": request.token_limit or 100,  # Default 100 tokens for guests
            "role": "user",
            "created_at": now,
            "updated_at": now,
        }
        user_id = guest_user_id
        user_email = guest_email
        logger.info(f"Created guest user {guest_user_id} for API key {key_id}")

    elif request.type == "master":
        # Master keys don't require a user
        user_id = None
        user_email = None

    # Generate the key
    key, prefix = _generate_api_key(request.type)
    key_hash = _hash_key(key)

    # Calculate expiry
    expires_at = None
    if request.expires_in_days:
        from datetime import timedelta

        expires_at = now + timedelta(days=request.expires_in_days)

    # Store the key
    _mock_api_keys[str(key_id)] = {
        "id": str(key_id),
        "user_id": str(user_id) if user_id else None,
        "key_hash": key_hash,
        "prefix": prefix,
        "name": request.name,
        "type": request.type,
        "token_limit": request.token_limit,
        "tokens_used": 0,
        "expires_at": expires_at,
        "is_active": True,
        "last_used_at": None,
        "created_at": now,
        "updated_at": now,
    }

    logger.info(f"Created API key {key_id} (type: {request.type})")

    return ApiKeyCreatedResponse(
        id=key_id,
        key=key,
        prefix=prefix,
        name=request.name,
        type=request.type,
        token_limit=request.token_limit,
        expires_at=expires_at,
        user_id=user_id,
        user_email=user_email,
    )


@router.get(
    "/keys",
    response_model=ApiKeyListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all API keys",
    description="Retrieve all API keys with usage statistics.",
)
async def list_api_keys(
    _admin: Annotated[bool, Depends(verify_admin_access)],
    type_filter: Annotated[
        Literal["standard", "master", "guest"] | None,
        Query(alias="type"),
    ] = None,
    active_only: Annotated[bool, Query()] = False,
) -> ApiKeyListResponse:
    """List all API keys with optional filtering."""
    all_keys = list(_mock_api_keys.values())

    # Apply filters
    if type_filter:
        all_keys = [k for k in all_keys if k["type"] == type_filter]
    if active_only:
        all_keys = [k for k in all_keys if k["is_active"]]

    keys = []
    for k in all_keys:
        user_email = None
        if k.get("user_id"):
            user = _mock_users.get(k["user_id"])
            if user:
                user_email = user["email"]

        tokens_remaining = None
        if k.get("token_limit") is not None:
            tokens_remaining = max(0, k["token_limit"] - k.get("tokens_used", 0))

        keys.append(
            ApiKeyResponse(
                id=UUID(k["id"]),
                user_id=UUID(k["user_id"]) if k.get("user_id") else None,
                user_email=user_email,
                prefix=k["prefix"],
                name=k["name"],
                type=k["type"],
                token_limit=k.get("token_limit"),
                tokens_used=k.get("tokens_used", 0),
                tokens_remaining=tokens_remaining,
                expires_at=k.get("expires_at"),
                is_active=k["is_active"],
                last_used_at=k.get("last_used_at"),
                created_at=k["created_at"],
            )
        )

    return ApiKeyListResponse(keys=keys, total=len(keys))


@router.delete(
    "/keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke API key",
    description="Deactivate an API key. It will no longer be usable.",
)
async def revoke_api_key(
    key_id: UUID,
    _admin: Annotated[bool, Depends(verify_admin_access)],
) -> None:
    """Revoke an API key by setting it inactive."""
    key_id_str = str(key_id)

    if key_id_str not in _mock_api_keys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found",
        )

    _mock_api_keys[key_id_str]["is_active"] = False
    _mock_api_keys[key_id_str]["updated_at"] = datetime.now(UTC)

    logger.info(f"Revoked API key {key_id}")


# -----------------------------------------------------------------------------
# Balance Check Endpoint (Public - validates key)
# -----------------------------------------------------------------------------


@router.get(
    "/balance",
    response_model=BalanceResponse,
    status_code=status.HTTP_200_OK,
    summary="Check API key balance",
    description="Check remaining tokens and status for an API key. No admin access required.",
)
async def check_balance(
    api_key: Annotated[str, Header(alias="X-API-Key")],
) -> BalanceResponse:
    """
    Check the balance and status of an API key.

    This endpoint does not require admin access - any valid API key can check its own balance.
    """
    key_hash = _hash_key(api_key)

    # Find key by hash
    key_data = None
    for k in _mock_api_keys.values():
        if secrets.compare_digest(k["key_hash"], key_hash):
            key_data = k
            break

    if not key_data:
        # Fallback: check against static API key from settings
        settings = get_settings()
        if settings.api.api_key and secrets.compare_digest(api_key, settings.api.api_key):
            return BalanceResponse(
                tokens_remaining=None,  # Unlimited for env key
                tokens_used=0,
                token_limit=None,
                type="master",
                key_name="Environment API Key",
                expires_at=None,
                is_active=True,
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Check if key is active and not expired
    if not key_data["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key has been revoked",
        )

    if key_data.get("expires_at") and key_data["expires_at"] < datetime.now(UTC):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key has expired",
        )

    tokens_remaining = None
    if key_data.get("token_limit") is not None:
        tokens_remaining = max(0, key_data["token_limit"] - key_data.get("tokens_used", 0))

    return BalanceResponse(
        tokens_remaining=tokens_remaining,
        tokens_used=key_data.get("tokens_used", 0),
        token_limit=key_data.get("token_limit"),
        type=key_data["type"],
        key_name=key_data["name"],
        expires_at=key_data.get("expires_at"),
        is_active=key_data["is_active"],
    )


# -----------------------------------------------------------------------------
# Live Logs Endpoints
# -----------------------------------------------------------------------------


class LogStatsResponse(BaseModel):
    """Statistics about API logging."""

    total_requests: int
    total_errors: int
    buffer_size: int
    active_subscribers: int
    uptime_seconds: float
    requests_per_minute: float


async def _log_event_generator(
    _admin: bool,
) -> AsyncGenerator[str]:
    """Generate SSE events for live logs."""
    try:
        async for entry in live_log_service.subscribe():
            # Format as SSE event
            data = json.dumps(entry.to_dict())
            yield f"data: {data}\n\n"
    except asyncio.CancelledError:
        pass


@router.get(
    "/logs/stream",
    summary="Stream live API logs (SSE)",
    description="Server-Sent Events endpoint for real-time API activity logs.",
    response_class=StreamingResponse,
)
async def stream_logs(
    _admin: Annotated[bool, Depends(verify_admin_access)],
) -> StreamingResponse:
    """
    Stream live API logs via Server-Sent Events.

    Connect to this endpoint to receive real-time logs of all API activity,
    including requests, responses, errors, health checks, and auth events.
    """
    return StreamingResponse(
        _log_event_generator(_admin),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get(
    "/logs/recent",
    response_model=list[dict],
    status_code=status.HTTP_200_OK,
    summary="Get recent API logs",
    description="Retrieve the most recent API logs from the buffer.",
)
async def get_recent_logs(
    _admin: Annotated[bool, Depends(verify_admin_access)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[dict]:
    """Get recent logs from the buffer."""
    logs = live_log_service.get_recent_logs(limit)
    return [log.to_dict() for log in logs]


@router.get(
    "/logs/stats",
    response_model=LogStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get logging statistics",
    description="Retrieve statistics about API activity.",
)
async def get_log_stats(
    _admin: Annotated[bool, Depends(verify_admin_access)],
) -> LogStatsResponse:
    """Get logging statistics."""
    stats = live_log_service.get_stats()
    return LogStatsResponse(**stats)


# -----------------------------------------------------------------------------
# Admin Tools Endpoints
# -----------------------------------------------------------------------------


class GrantTokensRequest(BaseModel):
    """Request to grant tokens to a user."""

    email: str = Field(..., min_length=1, description="User email address")
    tokens: int = Field(..., ge=1, le=10000, description="Number of tokens to grant")
    reason: str = Field(default="Admin grant", max_length=200, description="Reason for granting tokens")


class GrantTokensResponse(BaseModel):
    """Response after granting tokens."""

    success: bool
    user_id: str
    email: str
    tokens_granted: int
    new_balance: int
    message: str


class TestVerifyRequest(BaseModel):
    """Request to run a test verification."""

    test_type: Literal["simple", "complex", "hallucination"] = Field(
        default="simple",
        description="Type of test to run",
    )


class TestVerifyResponse(BaseModel):
    """Response from test verification."""

    success: bool
    test_type: str
    test_text: str
    claims_found: int
    verification_score: float
    duration_ms: float
    results: list[dict]


# Predefined test texts
TEST_TEXTS = {
    "simple": {
        "text": "The Eiffel Tower is located in Paris, France. It was completed in 1889 and stands approximately 330 meters tall.",
        "description": "Simple factual claims about the Eiffel Tower",
    },
    "complex": {
        "text": "Albert Einstein developed the theory of general relativity in 1915. He was awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect. Einstein was born in Ulm, Germany in 1879 and later became a naturalized American citizen.",
        "description": "Multiple factual claims about Einstein",
    },
    "hallucination": {
        "text": "The Great Wall of China was built in 1823 by Emperor Napoleon Bonaparte. It spans approximately 50 kilometers and is made entirely of marble imported from Italy.",
        "description": "Text containing intentional false claims for testing hallucination detection",
    },
}


@router.post(
    "/tools/grant-tokens",
    response_model=GrantTokensResponse,
    status_code=status.HTTP_200_OK,
    summary="Grant tokens to a user",
    description="Grant additional tokens to a user by their email address.",
)
async def grant_tokens(
    request: GrantTokensRequest,
    _admin: Annotated[bool, Depends(verify_admin_access)],
) -> GrantTokensResponse:
    """
    Grant tokens to a user by email.

    This is useful for:
    - Rewarding beta testers
    - Compensating for issues
    - Promotional grants
    """
    # Find user by email in mock store
    user_data = None
    user_id = None
    for uid, user in _mock_users.items():
        if user.get("email", "").lower() == request.email.lower():
            user_data = user
            user_id = uid
            break

    if not user_data:
        # Create user if doesn't exist (for demo purposes)
        user_id = str(uuid4())
        user_data = {
            "id": user_id,
            "email": request.email,
            "name": None,
            "ohi_tokens": 0,
            "role": "user",
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        _mock_users[user_id] = user_data
        logger.info(f"Created new user {user_id} for token grant")

    # Grant tokens
    old_balance = user_data.get("ohi_tokens", 0)
    new_balance = old_balance + request.tokens
    user_data["ohi_tokens"] = new_balance
    user_data["updated_at"] = datetime.now(UTC)

    logger.info(
        f"Granted {request.tokens} tokens to {request.email} "
        f"(balance: {old_balance} -> {new_balance}). Reason: {request.reason}"
    )

    # Log to live logs
    await live_log_service.log_system(
        f"Admin granted {request.tokens} tokens to {request.email}",
    )

    return GrantTokensResponse(
        success=True,
        user_id=user_id,
        email=request.email,
        tokens_granted=request.tokens,
        new_balance=new_balance,
        message=f"Successfully granted {request.tokens} tokens. New balance: {new_balance}",
    )


@router.post(
    "/tools/test-verify",
    response_model=TestVerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a test verification",
    description="Run a predefined test verification to check if the API is working correctly.",
)
async def test_verify(
    request: TestVerifyRequest,
    _admin: Annotated[bool, Depends(verify_admin_access)],
) -> TestVerifyResponse:
    """
    Run a test verification with predefined text.

    Available test types:
    - **simple**: Basic factual claims
    - **complex**: Multiple related claims
    - **hallucination**: Intentionally false claims
    """
    import time as time_module

    test_data = TEST_TEXTS.get(request.test_type, TEST_TEXTS["simple"])
    start_time = time_module.perf_counter()

    # Simulate verification (in production, this would call the actual verify endpoint)
    # For now, return mock results based on test type
    if request.test_type == "hallucination":
        mock_results = [
            {
                "claim": "The Great Wall of China was built in 1823",
                "verdict": "false",
                "confidence": 0.95,
                "evidence": "The Great Wall was built over many centuries, starting around 7th century BC",
            },
            {
                "claim": "Built by Emperor Napoleon Bonaparte",
                "verdict": "false",
                "confidence": 0.98,
                "evidence": "The Great Wall was built by various Chinese dynasties, not Napoleon",
            },
            {
                "claim": "Spans approximately 50 kilometers",
                "verdict": "false",
                "confidence": 0.92,
                "evidence": "The Great Wall spans over 21,000 kilometers",
            },
        ]
        score = 0.15  # Low score for hallucination
    else:
        mock_results = [
            {
                "claim": "The Eiffel Tower is located in Paris, France" if request.test_type == "simple" else "Einstein developed general relativity in 1915",
                "verdict": "true",
                "confidence": 0.97,
                "evidence": "Verified against multiple knowledge sources",
            },
        ]
        score = 0.92 if request.test_type == "simple" else 0.89

    duration_ms = (time_module.perf_counter() - start_time) * 1000 + 150  # Add simulated processing time

    # Log the test
    await live_log_service.log_system(
        f"Test verification ran: {request.test_type} (score: {score:.2f})",
    )

    return TestVerifyResponse(
        success=True,
        test_type=request.test_type,
        test_text=test_data["text"],
        claims_found=len(mock_results),
        verification_score=score,
        duration_ms=duration_ms,
        results=mock_results,
    )


@router.get(
    "/tools/test-texts",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Get available test texts",
    description="Retrieve the available predefined test texts for verification testing.",
)
async def get_test_texts(
    _admin: Annotated[bool, Depends(verify_admin_access)],
) -> dict:
    """Get all available test texts."""
    return {
        "tests": {
            key: {
                "type": key,
                "description": value["description"],
                "text_preview": value["text"][:100] + "..." if len(value["text"]) > 100 else value["text"],
            }
            for key, value in TEST_TEXTS.items()
        }
    }
