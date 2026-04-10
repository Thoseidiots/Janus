import { z } from "zod";
import { publicProcedure, router } from "../_core/trpc";
import { authenticateUser, createUser } from "../auth";
import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "../_core/cookies";

/**
 * Local authentication router
 * No external OAuth - simple username/password login
 */

export const authRouter = router({
  // Get current user from session
  me: publicProcedure.query(({ ctx }) => {
    return ctx.user || null;
  }),

  // Login with email and password
  login: publicProcedure
    .input(
      z.object({
        email: z.string().email(),
        password: z.string().min(1),
      })
    )
    .mutation(async ({ input, ctx }) => {
      const user = await authenticateUser(input.email, input.password);

      if (!user) {
        throw new Error("Invalid email or password");
      }

      // Set session cookie
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.cookie(COOKIE_NAME, user.id.toString(), cookieOptions);

      return {
        success: true,
        user: {
          id: user.id,
          email: user.email,
          name: user.name,
          role: user.role,
        },
      };
    }),

  // Register new user
  register: publicProcedure
    .input(
      z.object({
        email: z.string().email(),
        name: z.string().min(1),
        password: z.string().min(6),
      })
    )
    .mutation(async ({ input, ctx }) => {
      try {
        await createUser(input.email, input.name, input.password);

        // Auto-login after registration
        const user = await authenticateUser(input.email, input.password);
        if (user) {
          const cookieOptions = getSessionCookieOptions(ctx.req);
          ctx.res.cookie(COOKIE_NAME, user.id.toString(), cookieOptions);
        }

        return {
          success: true,
          message: "User created successfully",
        };
      } catch (error) {
        throw new Error(error instanceof Error ? error.message : "Registration failed");
      }
    }),

  // Logout
  logout: publicProcedure.mutation(({ ctx }) => {
    const cookieOptions = getSessionCookieOptions(ctx.req);
    ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
    return {
      success: true,
    };
  }),
});
