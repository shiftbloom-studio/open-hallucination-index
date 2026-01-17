import { pgTable, text, timestamp, uuid, boolean, integer, pgEnum } from 'drizzle-orm/pg-core';

// Enums
export const userRoleEnum = pgEnum('user_role', ['user', 'admin']);
export const apiKeyTypeEnum = pgEnum('api_key_type', ['standard', 'master', 'guest']);

export const users = pgTable('users', {
  id: uuid('id').primaryKey().defaultRandom(),
  email: text('email').notNull().unique(),
  name: text('name'),
  ohiTokens: integer('ohi_tokens').default(5).notNull(), // New users get 5 free tokens
  role: userRoleEnum('role').default('user').notNull(), // Admin role for dashboard access
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

// API Keys table for advanced key management
export const apiKeys = pgTable('api_keys', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id), // Nullable for system/guest keys
  keyHash: text('key_hash').notNull(), // SHA-256 hash of the actual key
  prefix: text('prefix').notNull(), // First 8 chars for display (e.g., "ohi_sk_ab")
  name: text('name').notNull(), // Friendly name (e.g., "Production API Key")
  type: apiKeyTypeEnum('type').default('standard').notNull(), // standard, master, guest
  tokenLimit: integer('token_limit'), // Nullable for unlimited
  tokensUsed: integer('tokens_used').default(0).notNull(), // Track usage
  expiresAt: timestamp('expires_at'), // Nullable for non-expiring keys
  isActive: boolean('is_active').default(true).notNull(),
  lastUsedAt: timestamp('last_used_at'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

export const hallucinations = pgTable('hallucinations', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id).notNull(),
  content: text('content').notNull(),
  source: text('source').notNull(),
  severity: text('severity').notNull(),
  verified: boolean('verified').default(false).notNull(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});
