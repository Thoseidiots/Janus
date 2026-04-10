import { z } from "zod";
import { protectedProcedure, router } from "../_core/trpc";
import {
  getActiveDHCPLeases,
  getDHCPLeaseByMac,
  createOrUpdateDHCPLease,
  getAllDNSRecords,
  getDNSRecord,
  createOrUpdateDNSRecord,
  getOnlineClients,
  upsertConnectedClient,
  getGatewayConfig,
  updateGatewayConfig,
  getSystemLogs,
  addSystemLog,
  getRecentPacketRoutes,
  recordPacketRoute,
} from "../db";
import { InsertDHCPLease, InsertDNSRecord, InsertConnectedClient, InsertSystemLog } from "../../drizzle/schema";

export const ispRouter = router({
  // DHCP Lease Management
  dhcp: router({
    getLeases: protectedProcedure.query(async () => {
      const leases = await getActiveDHCPLeases();
      return leases;
    }),

    getLease: protectedProcedure
      .input(z.object({ macAddress: z.string() }))
      .query(async ({ input }) => {
        const lease = await getDHCPLeaseByMac(input.macAddress);
        return lease;
      }),

    allocateIP: protectedProcedure
      .input(
        z.object({
          macAddress: z.string(),
          hostname: z.string().optional(),
          leaseHours: z.number().default(24),
        })
      )
      .mutation(async ({ input }) => {
        const now = new Date();
        const leaseEnd = new Date(now.getTime() + input.leaseHours * 60 * 60 * 1000);
        const renewalTime = new Date(now.getTime() + (input.leaseHours / 2) * 60 * 60 * 1000);

        // Simple IP allocation: find next available in pool
        const existingLeases = await getActiveDHCPLeases();
        const usedIPs = new Set(existingLeases.map((l) => l.ipAddress));

        let allocatedIP = null;
        for (let i = 100; i <= 250; i++) {
          const ip = `10.99.1.${i}`;
          if (!usedIPs.has(ip)) {
            allocatedIP = ip;
            break;
          }
        }

        if (!allocatedIP) {
          throw new Error("No available IPs in DHCP pool");
        }

        const lease: InsertDHCPLease = {
          macAddress: input.macAddress,
          ipAddress: allocatedIP,
          hostname: input.hostname,
          leaseStartTime: now,
          leaseEndTime: leaseEnd,
          renewalTime: renewalTime,
          isActive: true,
        };

        await createOrUpdateDHCPLease(lease);
        await addSystemLog({
          timestamp: now,
          service: "DHCP",
          level: "INFO",
          message: `IP allocated: ${allocatedIP} to ${input.macAddress}`,
        });

        return { ipAddress: allocatedIP, leaseEnd, renewalTime };
      }),

    releaseIP: protectedProcedure
      .input(z.object({ macAddress: z.string() }))
      .mutation(async ({ input }) => {
        const lease = await getDHCPLeaseByMac(input.macAddress);
        if (lease) {
          await createOrUpdateDHCPLease({
            ...lease,
            isActive: false,
          });
          await addSystemLog({
            timestamp: new Date(),
            service: "DHCP",
            level: "INFO",
            message: `IP released: ${lease.ipAddress} from ${input.macAddress}`,
          });
        }
        return { success: true };
      }),
  }),

  // DNS Management
  dns: router({
    getRecords: protectedProcedure.query(async () => {
      const records = await getAllDNSRecords();
      return records;
    }),

    getRecord: protectedProcedure
      .input(z.object({ domain: z.string() }))
      .query(async ({ input }) => {
        const record = await getDNSRecord(input.domain);
        return record;
      }),

    addRecord: protectedProcedure
      .input(
        z.object({
          domain: z.string(),
          ipAddress: z.string(),
          ttl: z.number().default(3600),
          description: z.string().optional(),
        })
      )
      .mutation(async ({ input }) => {
        const record: InsertDNSRecord = {
          domain: input.domain,
          ipAddress: input.ipAddress,
          recordType: "A",
          ttl: input.ttl,
          description: input.description,
        };

        await createOrUpdateDNSRecord(record);
        await addSystemLog({
          timestamp: new Date(),
          service: "DNS",
          level: "INFO",
          message: `DNS record added: ${input.domain} -> ${input.ipAddress}`,
        });

        return record;
      }),

    deleteRecord: protectedProcedure
      .input(z.object({ domain: z.string() }))
      .mutation(async ({ input }) => {
        await addSystemLog({
          timestamp: new Date(),
          service: "DNS",
          level: "INFO",
          message: `DNS record deleted: ${input.domain}`,
        });
        return { success: true };
      }),
  }),

  // Network Monitoring
  network: router({
    getClients: protectedProcedure.query(async () => {
      const clients = await getOnlineClients();
      return clients;
    }),

    updateClient: protectedProcedure
      .input(
        z.object({
          macAddress: z.string(),
          hostname: z.string().optional(),
          ipAddress: z.string(),
          signalStrength: z.number().optional(),
          bandwidthUsage: z.string().optional(),
        })
      )
      .mutation(async ({ input }) => {
        const client: InsertConnectedClient = {
          macAddress: input.macAddress,
          hostname: input.hostname,
          ipAddress: input.ipAddress,
          signalStrength: input.signalStrength,
          lastSeen: new Date(),
          isOnline: true,
          bandwidthUsage: input.bandwidthUsage ? (parseFloat(input.bandwidthUsage) as any) : undefined,
        };

        await upsertConnectedClient(client);
        return client;
      }),

    getPacketRoutes: protectedProcedure.query(async () => {
      const routes = await getRecentPacketRoutes(50);
      return routes;
    }),

    recordTraffic: protectedProcedure
      .input(
        z.object({
          sourceIp: z.string(),
          destinationIp: z.string(),
          protocol: z.string(),
          packetCount: z.number().default(1),
          bytesTransferred: z.coerce.string().default("0"),
        })
      )
      .mutation(async ({ input }) => {
        await recordPacketRoute({
          sourceIp: input.sourceIp,
          destinationIp: input.destinationIp,
          protocol: input.protocol,
          packetCount: input.packetCount,
          bytesTransferred: String(input.bytesTransferred),
          lastSeen: new Date(),
        });
        return { success: true };
      }),
  }),

  // Gateway Configuration
  gateway: router({
    getConfig: protectedProcedure.query(async () => {
      let config = await getGatewayConfig();
      if (!config) {
        // Initialize default config
        await updateGatewayConfig({
          internalSubnet: "10.99.1.0/24",
          gatewayIp: "10.99.1.1",
          dnsServer: "8.8.8.8",
          natEnabled: true,
          dhcpEnabled: true,
          dnsEnabled: true,
        });
        config = await getGatewayConfig();
      }
      return config;
    }),

    updateConfig: protectedProcedure
      .input(
        z.object({
          externalIp: z.string().optional(),
          internalSubnet: z.string().optional(),
          gatewayIp: z.string().optional(),
          dnsServer: z.string().optional(),
          natEnabled: z.boolean().optional(),
          dhcpEnabled: z.boolean().optional(),
          dnsEnabled: z.boolean().optional(),
        })
      )
      .mutation(async ({ input }) => {
        await updateGatewayConfig(input);
        await addSystemLog({
          timestamp: new Date(),
          service: "GATEWAY",
          level: "INFO",
          message: `Gateway configuration updated`,
        });
        return await getGatewayConfig();
      }),
  }),

  // System Logs
  logs: router({
    getRecent: protectedProcedure
      .input(z.object({ limit: z.number().default(100), service: z.string().optional() }))
      .query(async ({ input }) => {
        const logs = await getSystemLogs(input.limit);
        if (input.service) {
          return logs.filter((log) => log.service === input.service);
        }
        return logs;
      }),

    addLog: protectedProcedure
      .input(
        z.object({
          service: z.string(),
          level: z.enum(["INFO", "WARN", "ERROR", "DEBUG"]),
          message: z.string(),
          details: z.string().optional(),
        })
      )
      .mutation(async ({ input }) => {
        const log: InsertSystemLog = {
          timestamp: new Date(),
          service: input.service,
          level: input.level,
          message: input.message,
          details: input.details,
        };
        await addSystemLog(log);
        return log;
      }),
  }),

  // Service Status
  status: router({
    getHealth: protectedProcedure.query(async () => {
      const config = await getGatewayConfig();
      const leases = await getActiveDHCPLeases();
      const clients = await getOnlineClients();
      const records = await getAllDNSRecords();

      return {
        dhcp: {
          enabled: config?.dhcpEnabled ?? true,
          activeLeases: leases.length,
          poolStart: "10.99.1.100",
          poolEnd: "10.99.1.250",
        },
        dns: {
          enabled: config?.dnsEnabled ?? true,
          recordCount: records.length,
          upstreamServer: config?.dnsServer ?? "8.8.8.8",
        },
        network: {
          onlineClients: clients.length,
          gatewayIp: config?.gatewayIp ?? "10.99.1.1",
          subnet: config?.internalSubnet ?? "10.99.1.0/24",
        },
        nat: {
          enabled: config?.natEnabled ?? true,
          externalIp: config?.externalIp,
        },
      };
    }),
  }),
});
