# Implementation Plan - Admin Dashboard & API Enhancements

This plan outlines the steps to implement a comprehensive Admin Dashboard, upgrade the API for reusability, and implement advanced API key management (User, Master, Guest) with Supabase integration.

## User Review Required

> [!IMPORTANT]
> **Database Schema Changes**: This plan involves modifying the `users` table and creating a new `api_keys` table. Please ensure running migrations is safe in your environment (backup recommended).

> [!WARNING]
> **API Authentication**: The current static `API_KEY` in settings will be deprecated in favor of database-backed API keys. We will migrate the current system to support both temporarily or fully switch. **Decision**: We will try to fully switch to DB keys, but keep the env key as a fallback "Master" key if the DB is empty or for bootstrapping.

## Proposed Changes

### Database Layer (`src/frontend/src/lib/db`)

#### [MODIFY] [schema.ts](file:///<home>/Documents/shiftbloom/git/open-hallucination-index/src/frontend/src/lib/db/schema.ts)
- Add `role` enum or `isAdmin` boolean to `users` table.
- Create `api_keys` table with fields:
    - `id` (uuid, pk)
    - `userId` (uuid, fk to users, nullable for system keys if needed, but preferred linked to an admin)
    - `keyHash` (text, for security)
    - `prefix` (text, for display)
    - `name` (text, friendly name)
    - `type` (enum: 'standard', 'master', 'guest')
    - `tokenLimit` (int, nullable for unlimited)
    - `expiresAt` (timestamp)
    - `isActive` (boolean)
    - `createdAt`, `updatedAt`

### Backend API (`src/api`)

#### [NEW] `src/api/server/routes/admin.py`
- Endpoints for admin management:
    - `GET /admin/users`: List all users (paginated).
    - `POST /admin/keys`: Create new API keys (handles Guest user creation logic).
    - `GET /admin/keys`: List all API keys.
    - `DELETE /admin/keys/{id}`: Revoke keys.

#### [MODIFY] `src/api/server/app.py`
- Update `verify_api_key` dependency to:
    - Extract key from header.
    - Hash it.
    - Check against `api_keys` table.
    - Validate `isActive`, `expiresAt`, and `tokenLimit`.
    - Setup "Live Logs" broadcasting (using FastAPI `WebSocket` or background tasks pushing to a queue/stream).

#### [MODIFY] `src/api/server/routes/verify.py` & `track.py`
- Ensure standardized JSON responses for better developer experience.
- Include usage headers (`X-Tokens-Remaining`).

### Frontend (`src/frontend`)

#### [NEW] `src/frontend/src/app/admin/page.tsx`
- Admin Dashboard UI using existing components + Shadcn UI.
- Sections:
    - **Overview**: Live anonymous logs (scrollable terminal-like view).
    - **Users**: Table of users with "Make Admin" toggle.
    - **API Keys**: List of keys. Form to create new keys:
        - Type: User (select user), Master, Guest.
        - Token Limit input.
        - "Generate" button -> Displays key once.

#### [NEW] `src/frontend/src/app/admin/layout.tsx`
- Project route: Check if current user is Admin. Redirect if not.

### MCP Server (`src/ohi-mcp-server`)

#### [MODIFY] `src/ohi-mcp-server/src/index.ts`
- Add tool: `ohi_check_balance`
    - Input: `api_key` (string)
    - Output: `{ "tokens_remaining": 123, "type": "guest" }`
    - Logic: Calls OHI API `/api/v1/user/balance` (new endpoint).

## Verification Plan

### Automated Tests
- **API Tests**:
    - Test Key Creation (Master, User, Guest).
    - Test Key Verification (Valid, Invalid, Expired, Rate Limited).
    - Test Admin Routes (Authz check).
    - `pytest src/api/tests/test_auth.py` (New test file).
- **Frontend Tests**:
    - `playwright`: Log in as Admin -> Verify Dashboard loads.
    - `playwright`: Create Guest Key -> Verify "Guest" user appears in User list.

### Manual Verification
1.  **Bootstrapping**: Manually insert an Admin user into DB (or use a setup script).
2.  **Dashboard**: Login as Admin, go to `/admin`.
3.  **Live Logs**: Open Dashboard, make API req in separate terminal, watch logs appear.
4.  **Guest Key**: Create Guest Key with 5 tokens. Use it 6 times. Verify 6th fails.
