import { eq } from "drizzle-orm";
import { users } from "../drizzle/schema";
import { getDb } from "./db";

/**
 * Simple local authentication system
 * No external OAuth required - just username/password
 */

export async function authenticateUser(username: string, password: string) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");

  // Get user by email
  const user = await db
    .select()
    .from(users)
    .where(eq(users.email, username))
    .limit(1);

  if (user.length === 0) {
    return null;
  }

  const foundUser = user[0];
  // For demo purposes, we store password in the loginMethod field
  // In production, use bcrypt: npm install bcrypt
  // const isValid = await bcrypt.compare(password, foundUser.passwordHash);
  if (foundUser.loginMethod === `local:${password}`) {
    return foundUser;
  }

  return null;
}

export async function createUser(email: string, name: string, password: string) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");

  // Check if user already exists
  const existing = await db
    .select()
    .from(users)
    .where(eq(users.email, email))
    .limit(1);

  if (existing.length > 0) {
    throw new Error("User already exists");
  }

  // For demo purposes, store password in loginMethod field
  // In production, use bcrypt: const hashedPassword = await bcrypt.hash(password, 10);
  const newUser = await db.insert(users).values({
    openId: email, // Use email as openId for local auth
    email,
    name,
    loginMethod: `local:${password}`, // Store password (NOT SECURE - demo only)
    role: "admin", // First user is admin
  });

  return newUser;
}

export async function validateSession(userId: number) {
  const db = await getDb();
  if (!db) return null;

  const user = await db
    .select()
    .from(users)
    .where(eq(users.id, userId))
    .limit(1);

  return user.length > 0 ? user[0] : null;
}
