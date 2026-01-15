import { describe, it, expect } from 'vitest';
import { users, hallucinations } from '@/lib/db/schema';

describe('Database Schema', () => {
  describe('users table', () => {
    it('should be defined', () => {
      expect(users).toBeDefined();
    });

    it('should have id column', () => {
      expect(users.id).toBeDefined();
      expect(users.id.name).toBe('id');
    });

    it('should have email column', () => {
      expect(users.email).toBeDefined();
      expect(users.email.name).toBe('email');
    });

    it('should have name column', () => {
      expect(users.name).toBeDefined();
      expect(users.name.name).toBe('name');
    });

    it('should have ohiTokens column', () => {
      expect(users.ohiTokens).toBeDefined();
      expect(users.ohiTokens.name).toBe('ohi_tokens');
    });

    it('should have createdAt column', () => {
      expect(users.createdAt).toBeDefined();
      expect(users.createdAt.name).toBe('created_at');
    });

    it('should have updatedAt column', () => {
      expect(users.updatedAt).toBeDefined();
      expect(users.updatedAt.name).toBe('updated_at');
    });
  });

  describe('hallucinations table', () => {
    it('should be defined', () => {
      expect(hallucinations).toBeDefined();
    });

    it('should have id column', () => {
      expect(hallucinations.id).toBeDefined();
      expect(hallucinations.id.name).toBe('id');
    });

    it('should have userId column with foreign key', () => {
      expect(hallucinations.userId).toBeDefined();
      expect(hallucinations.userId.name).toBe('user_id');
    });

    it('should have content column', () => {
      expect(hallucinations.content).toBeDefined();
      expect(hallucinations.content.name).toBe('content');
    });

    it('should have source column', () => {
      expect(hallucinations.source).toBeDefined();
      expect(hallucinations.source.name).toBe('source');
    });

    it('should have severity column', () => {
      expect(hallucinations.severity).toBeDefined();
      expect(hallucinations.severity.name).toBe('severity');
    });

    it('should have verified column', () => {
      expect(hallucinations.verified).toBeDefined();
      expect(hallucinations.verified.name).toBe('verified');
    });

    it('should have createdAt column', () => {
      expect(hallucinations.createdAt).toBeDefined();
      expect(hallucinations.createdAt.name).toBe('created_at');
    });

    it('should have updatedAt column', () => {
      expect(hallucinations.updatedAt).toBeDefined();
      expect(hallucinations.updatedAt.name).toBe('updated_at');
    });
  });
});
