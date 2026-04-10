/**
 * Bcrypt Password Hashing Module
 * Provides secure password hashing and verification using bcrypt
 * 
 * Usage:
 *   const hashedPassword = await hashPassword(plainPassword);
 *   const isValid = await verifyPassword(plainPassword, hashedPassword);
 */

import bcrypt from 'bcrypt';

// Bcrypt salt rounds - higher = more secure but slower
// 10 is a good balance between security and performance
const SALT_ROUNDS = 10;

/**
 * Hash a plain text password using bcrypt
 * @param password - Plain text password to hash
 * @returns Promise<string> - Hashed password
 */
export async function hashPassword(password: string): Promise<string> {
  if (!password || password.length === 0) {
    throw new Error('Password cannot be empty');
  }
  
  if (password.length > 72) {
    throw new Error('Password is too long (max 72 characters for bcrypt)');
  }
  
  try {
    const hashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
    return hashedPassword;
  } catch (error) {
    throw new Error(`Failed to hash password: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

/**
 * Verify a plain text password against a bcrypt hash
 * @param password - Plain text password to verify
 * @param hash - Bcrypt hash to compare against
 * @returns Promise<boolean> - True if password matches, false otherwise
 */
export async function verifyPassword(password: string, hash: string): Promise<boolean> {
  if (!password || !hash) {
    return false;
  }
  
  try {
    const isValid = await bcrypt.compare(password, hash);
    return isValid;
  } catch (error) {
    // Bcrypt comparison errors (e.g., invalid hash format) should return false
    console.error('Password verification error:', error);
    return false;
  }
}

/**
 * Check if a password hash is in the old plaintext format (for migration)
 * @param loginMethod - The loginMethod field value
 * @returns boolean - True if it's in old format (local:password)
 */
export function isOldPasswordFormat(loginMethod: string): boolean {
  return loginMethod.startsWith('local:') && !loginMethod.startsWith('local:$2');
}

/**
 * Extract password from old format for migration
 * @param loginMethod - The loginMethod field value in old format
 * @returns string - The plain text password
 */
export function extractOldPassword(loginMethod: string): string {
  if (!isOldPasswordFormat(loginMethod)) {
    throw new Error('Not in old password format');
  }
  return loginMethod.substring(6); // Remove 'local:' prefix
}
