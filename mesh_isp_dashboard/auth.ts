import { router } from "./_core/trpc";
import { ispRouter } from "./routers/isp";
import { authRouter } from "./routers/auth";

export const appRouter = router({
  // Local authentication (no external OAuth required)
  auth: authRouter,

  // ISP Services
  isp: ispRouter,
});

export type AppRouter = typeof appRouter;
