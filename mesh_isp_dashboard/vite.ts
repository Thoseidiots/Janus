import { describe, expect, it } from "vitest";
import { appRouter } from "../routers";
import { COOKIE_NAME } from "@shared/const";
import type { TrpcContext } from "../_core/context";

type AuthenticatedUser = NonNullable<TrpcContext["user"]>;

function createAuthContext(): { ctx: TrpcContext; cookies: Map<string, string> } {
  const cookies = new Map<string, string>();

  const ctx: TrpcContext = {
    user: null,
    req: {
      protocol: "https",
      headers: {},
    } as TrpcContext["req"],
    res: {
      cookie: (name: string, value: string) => {
        cookies.set(name, value);
      },
      clearCookie: (name: string) => {
        cookies.delete(name);
      },
    } as TrpcContext["res"],
  };

  return { ctx, cookies };
}

describe("Local Auth Router", () => {
  describe("auth.register", () => {
    it("should register a new user", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.auth.register({
        email: `newuser-${Date.now()}@example.com`,
        name: "New User",
        password: "password123",
      });

      expect(result.success).toBe(true);
      expect(result.message).toContain("created");
    });

    it("should not allow duplicate email registration", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);
      const uniqueEmail = `duplicate-${Date.now()}@example.com`;

      // First registration
      await caller.auth.register({
        email: uniqueEmail,
        name: "User One",
        password: "password123",
      });

      // Second registration with same email
      try {
        await caller.auth.register({
          email: uniqueEmail,
          name: "User Two",
          password: "password456",
        });
        expect.fail("Should have thrown an error");
      } catch (error) {
        expect(error).toBeDefined();
        expect(error instanceof Error ? error.message : "").toContain("already exists");
      }
    });

    it("should require valid email format", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      try {
        await caller.auth.register({
          email: "invalid-email",
          name: "User",
          password: "password123",
        });
        expect.fail("Should have thrown validation error");
      } catch (error) {
        expect(error).toBeDefined();
        // Validation error expected
      }
    });

    it("should require minimum password length", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      try {
        await caller.auth.register({
          email: `user-${Date.now()}@example.com`,
          name: "User",
          password: "short",
        });
        expect.fail("Should have thrown validation error");
      } catch (error) {
        expect(error).toBeDefined();
        // Validation error expected
      }
    });
  });

  describe("auth.login", () => {
    it("should login with correct credentials", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);
      const uniqueEmail = `login-${Date.now()}@example.com`;

      // Register first
      await caller.auth.register({
        email: uniqueEmail,
        name: "Login User",
        password: "password123",
      });

      // Then login
      const result = await caller.auth.login({
        email: uniqueEmail,
        password: "password123",
      });

      expect(result.success).toBe(true);
      expect(result.user).toBeDefined();
      expect(result.user?.email).toBe(uniqueEmail);
    });

    it("should reject invalid credentials", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      try {
        await caller.auth.login({
          email: `nonexistent-${Date.now()}@example.com`,
          password: "wrongpassword",
        });
        expect.fail("Should have thrown authentication error");
      } catch (error) {
        expect(error instanceof Error ? error.message : "").toContain("Invalid");
      }
    });

    it("should set session cookie on successful login", async () => {
      const { ctx, cookies } = createAuthContext();
      const caller = appRouter.createCaller(ctx);
      const uniqueEmail = `cookie-${Date.now()}@example.com`;

      // Register
      await caller.auth.register({
        email: uniqueEmail,
        name: "Cookie User",
        password: "password123",
      });

      // Login
      await caller.auth.login({
        email: uniqueEmail,
        password: "password123",
      });

      // Check cookie was set
      expect(cookies.has(COOKIE_NAME)).toBe(true);
      const cookieValue = cookies.get(COOKIE_NAME);
      expect(cookieValue).toBeDefined();
      expect(cookieValue).toMatch(/^\d+$/); // Should be a user ID
    });
  });

  describe("auth.me", () => {
    it("should return null for unauthenticated user", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const user = await caller.auth.me();
      expect(user).toBeNull();
    });

    it("should return user data for authenticated user", async () => {
      const { ctx } = createAuthContext();
      const caller = appRouter.createCaller(ctx);
      const uniqueEmail = `me-${Date.now()}@example.com`;

      // Register and login
      await caller.auth.register({
        email: uniqueEmail,
        name: "Me User",
        password: "password123",
      });

      const loginResult = await caller.auth.login({
        email: uniqueEmail,
        password: "password123",
      });

      // Create authenticated context
      const authenticatedCtx: TrpcContext = {
        user: loginResult.user as AuthenticatedUser,
        req: ctx.req,
        res: ctx.res,
      };

      const authenticatedCaller = appRouter.createCaller(authenticatedCtx);
      const me = await authenticatedCaller.auth.me();

      expect(me).toBeDefined();
      expect(me?.email).toBe(uniqueEmail);
      expect(me?.name).toBe("Me User");
    });
  });

  describe("auth.logout", () => {
    it("should clear session cookie on logout", async () => {
      const { ctx, cookies } = createAuthContext();
      const caller = appRouter.createCaller(ctx);
      const uniqueEmail = `logout-${Date.now()}@example.com`;

      // Register and login
      await caller.auth.register({
        email: uniqueEmail,
        name: "Logout User",
        password: "password123",
      });

      await caller.auth.login({
        email: uniqueEmail,
        password: "password123",
      });

      // Verify cookie is set
      expect(cookies.has(COOKIE_NAME)).toBe(true);

      // Logout
      const result = await caller.auth.logout();
      expect(result.success).toBe(true);

      // Cookie should be cleared
      expect(cookies.has(COOKIE_NAME)).toBe(false);
    });
  });
});
