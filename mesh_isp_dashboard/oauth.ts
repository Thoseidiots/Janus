import { describe, expect, it, beforeEach, vi } from "vitest";
import { appRouter } from "../routers";
import type { TrpcContext } from "../_core/context";

type AuthenticatedUser = NonNullable<TrpcContext["user"]>;

function createAuthContext(): TrpcContext {
  const user: AuthenticatedUser = {
    id: 1,
    openId: "test-user",
    email: "test@example.com",
    name: "Test User",
    loginMethod: "manus",
    role: "user",
    createdAt: new Date(),
    updatedAt: new Date(),
    lastSignedIn: new Date(),
  };

  const ctx: TrpcContext = {
    user,
    req: {
      protocol: "https",
      headers: {},
    } as TrpcContext["req"],
    res: {} as TrpcContext["res"],
  };

  return ctx;
}

describe("ISP Router", () => {
  describe("DHCP", () => {
    it("should get active leases", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const leases = await caller.isp.dhcp.getLeases();
      expect(Array.isArray(leases)).toBe(true);
    });

    it("should allocate an IP address", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.isp.dhcp.allocateIP({
        macAddress: "00:11:22:33:44:55",
        hostname: "test-device",
        leaseHours: 24,
      });

      expect(result).toHaveProperty("ipAddress");
      expect(result).toHaveProperty("leaseEnd");
      expect(result).toHaveProperty("renewalTime");
      expect(result.ipAddress).toMatch(/^10\.99\.1\.\d+$/);
    });

    it("should get a specific lease by MAC address", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      // First allocate an IP
      const allocated = await caller.isp.dhcp.allocateIP({
        macAddress: "00:11:22:33:44:66",
        leaseHours: 24,
      });

      // Then retrieve it
      const lease = await caller.isp.dhcp.getLease({
        macAddress: "00:11:22:33:44:66",
      });

      expect(lease).toBeDefined();
      expect(lease?.macAddress).toBe("00:11:22:33:44:66");
      expect(lease?.ipAddress).toBe(allocated.ipAddress);
    });

    it("should release an IP address", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      // First allocate an IP
      await caller.isp.dhcp.allocateIP({
        macAddress: "00:11:22:33:44:77",
        leaseHours: 24,
      });

      // Then release it
      const result = await caller.isp.dhcp.releaseIP({
        macAddress: "00:11:22:33:44:77",
      });

      expect(result.success).toBe(true);
    });
  });

  describe("DNS", () => {
    it("should get all DNS records", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const records = await caller.isp.dns.getRecords();
      expect(Array.isArray(records)).toBe(true);
    });

    it("should add a DNS record", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.isp.dns.addRecord({
        domain: "test.mesh",
        ipAddress: "10.99.1.5",
        ttl: 3600,
        description: "Test record",
      });

      expect(result).toHaveProperty("domain");
      expect(result.domain).toBe("test.mesh");
      expect(result.ipAddress).toBe("10.99.1.5");
      expect(result.ttl).toBe(3600);
    });

    it("should get a specific DNS record", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      // First add a record
      await caller.isp.dns.addRecord({
        domain: "dashboard.mesh",
        ipAddress: "10.99.1.1",
        ttl: 3600,
      });

      // Then retrieve it
      const record = await caller.isp.dns.getRecord({
        domain: "dashboard.mesh",
      });

      expect(record).toBeDefined();
      expect(record?.domain).toBe("dashboard.mesh");
      expect(record?.ipAddress).toBe("10.99.1.1");
    });

    it("should delete a DNS record", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.isp.dns.deleteRecord({
        domain: "test.mesh",
      });

      expect(result.success).toBe(true);
    });
  });

  describe("Network", () => {
    it("should get connected clients", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const clients = await caller.isp.network.getClients();
      expect(Array.isArray(clients)).toBe(true);
    });

    it("should update client information", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.isp.network.updateClient({
        macAddress: "aa:bb:cc:dd:ee:ff",
        hostname: "laptop",
        ipAddress: "10.99.1.100",
        signalStrength: 85,
      });

      expect(result).toHaveProperty("macAddress");
      expect(result.macAddress).toBe("aa:bb:cc:dd:ee:ff");
      expect(result.hostname).toBe("laptop");
      expect(result.ipAddress).toBe("10.99.1.100");
    });

    it("should get packet routes", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const routes = await caller.isp.network.getPacketRoutes();
      expect(Array.isArray(routes)).toBe(true);
    });

    it("should record traffic", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.isp.network.recordTraffic({
        sourceIp: "10.99.1.100",
        destinationIp: "8.8.8.8",
        protocol: "TCP",
        packetCount: 10,
        bytesTransferred: "5120",
      });

      expect(result.success).toBe(true);
    });
  });

  describe("Gateway", () => {
    it("should get gateway configuration", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const config = await caller.isp.gateway.getConfig();
      expect(config).toBeDefined();
      expect(config).toHaveProperty("gatewayIp");
      expect(config).toHaveProperty("internalSubnet");
      expect(config).toHaveProperty("dnsServer");
    });

    it("should update gateway configuration", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.isp.gateway.updateConfig({
        externalIp: "203.0.113.10",
        dnsServer: "1.1.1.1",
        natEnabled: true,
      });

      expect(result).toBeDefined();
      expect(result?.externalIp).toBe("203.0.113.10");
      expect(result?.dnsServer).toBe("1.1.1.1");
    });
  });

  describe("Logs", () => {
    it("should get recent system logs", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const logs = await caller.isp.logs.getRecent({ limit: 10 });
      expect(Array.isArray(logs)).toBe(true);
    });

    it("should add a system log", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const result = await caller.isp.logs.addLog({
        service: "TEST",
        level: "INFO",
        message: "Test log message",
        details: "Test details",
      });

      expect(result).toHaveProperty("service");
      expect(result.service).toBe("TEST");
      expect(result.message).toBe("Test log message");
    });

    it("should filter logs by service", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      // Add a log
      await caller.isp.logs.addLog({
        service: "DHCP",
        level: "INFO",
        message: "DHCP test",
      });

      // Get logs filtered by service
      const logs = await caller.isp.logs.getRecent({
        limit: 50,
        service: "DHCP",
      });

      expect(Array.isArray(logs)).toBe(true);
      logs.forEach((log) => {
        expect(log.service).toBe("DHCP");
      });
    });
  });

  describe("Status", () => {
    it("should get system health status", async () => {
      const ctx = createAuthContext();
      const caller = appRouter.createCaller(ctx);

      const health = await caller.isp.status.getHealth();

      expect(health).toHaveProperty("dhcp");
      expect(health).toHaveProperty("dns");
      expect(health).toHaveProperty("network");
      expect(health).toHaveProperty("nat");

      expect(health.dhcp).toHaveProperty("enabled");
      expect(health.dhcp).toHaveProperty("activeLeases");
      expect(health.dhcp).toHaveProperty("poolStart");
      expect(health.dhcp).toHaveProperty("poolEnd");

      expect(health.dns).toHaveProperty("enabled");
      expect(health.dns).toHaveProperty("recordCount");
      expect(health.dns).toHaveProperty("upstreamServer");

      expect(health.network).toHaveProperty("onlineClients");
      expect(health.network).toHaveProperty("gatewayIp");
      expect(health.network).toHaveProperty("subnet");

      expect(health.nat).toHaveProperty("enabled");
      expect(health.nat).toHaveProperty("externalIp");
    });
  });
});
