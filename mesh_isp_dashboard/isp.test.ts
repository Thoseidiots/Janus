import { eq, desc } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { InsertUser, users, dhcpLeases, dnsRecords, connectedClients, systemLogs, gatewayConfig, packetRoutes, InsertDHCPLease, InsertDNSRecord, InsertConnectedClient, InsertSystemLog, InsertGatewayConfig, InsertPacketRoute } from "../drizzle/schema";
import { ENV } from './_core/env';

let _db: ReturnType<typeof drizzle> | null = null;

// Lazily create the drizzle instance so local tooling can run without a DB.
export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) {
    throw new Error("User openId is required for upsert");
  }

  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot upsert user: database not available");
    return;
  }

  try {
    const values: InsertUser = {
      openId: user.openId,
    };
    const updateSet: Record<string, unknown> = {};

    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];

    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };

    textFields.forEach(assignNullable);

    if (user.lastSignedIn !== undefined) {
      values.lastSignedIn = user.lastSignedIn;
      updateSet.lastSignedIn = user.lastSignedIn;
    }
    if (user.role !== undefined) {
      values.role = user.role;
      updateSet.role = user.role;
    } else if (user.openId === ENV.ownerOpenId) {
      values.role = 'admin';
      updateSet.role = 'admin';
    }

    if (!values.lastSignedIn) {
      values.lastSignedIn = new Date();
    }

    if (Object.keys(updateSet).length === 0) {
      updateSet.lastSignedIn = new Date();
    }

    await db.insert(users).values(values).onDuplicateKeyUpdate({
      set: updateSet,
    });
  } catch (error) {
    console.error("[Database] Failed to upsert user:", error);
    throw error;
  }
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot get user: database not available");
    return undefined;
  }

  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);

  return result.length > 0 ? result[0] : undefined;
}

// DHCP Lease queries
export async function createOrUpdateDHCPLease(lease: InsertDHCPLease) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  const existing = await db.select().from(dhcpLeases).where(eq(dhcpLeases.macAddress, lease.macAddress)).limit(1);
  
  if (existing.length > 0) {
    await db.update(dhcpLeases).set(lease).where(eq(dhcpLeases.macAddress, lease.macAddress));
  } else {
    await db.insert(dhcpLeases).values(lease);
  }
}

export async function getActiveDHCPLeases() {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(dhcpLeases).where(eq(dhcpLeases.isActive, true));
}

export async function getDHCPLeaseByMac(macAddress: string) {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(dhcpLeases).where(eq(dhcpLeases.macAddress, macAddress)).limit(1);
  return result[0];
}

// DNS Record queries
export async function createOrUpdateDNSRecord(record: InsertDNSRecord) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  const existing = await db.select().from(dnsRecords).where(eq(dnsRecords.domain, record.domain)).limit(1);
  
  if (existing.length > 0) {
    await db.update(dnsRecords).set(record).where(eq(dnsRecords.domain, record.domain));
  } else {
    await db.insert(dnsRecords).values(record);
  }
}

export async function getAllDNSRecords() {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(dnsRecords);
}

export async function getDNSRecord(domain: string) {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(dnsRecords).where(eq(dnsRecords.domain, domain)).limit(1);
  return result[0];
}

// Connected Clients queries
export async function upsertConnectedClient(client: InsertConnectedClient) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  const existing = await db.select().from(connectedClients).where(eq(connectedClients.macAddress, client.macAddress)).limit(1);
  
  if (existing.length > 0) {
    await db.update(connectedClients).set(client).where(eq(connectedClients.macAddress, client.macAddress));
  } else {
    await db.insert(connectedClients).values(client);
  }
}

export async function getOnlineClients() {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(connectedClients).where(eq(connectedClients.isOnline, true));
}

// System Logs queries
export async function addSystemLog(log: InsertSystemLog) {
  const db = await getDb();
  if (!db) return;
  await db.insert(systemLogs).values(log);
}

export async function getSystemLogs(limit = 100) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(systemLogs).orderBy(desc(systemLogs.timestamp)).limit(limit);
}

// Gateway Config queries
export async function getGatewayConfig() {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(gatewayConfig).limit(1);
  return result[0];
}

export async function updateGatewayConfig(config: Partial<InsertGatewayConfig>) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  const existing = await getGatewayConfig();
  if (existing) {
    await db.update(gatewayConfig).set(config).where(eq(gatewayConfig.id, existing.id));
  } else {
    await db.insert(gatewayConfig).values(config as InsertGatewayConfig);
  }
}

// Packet Routes queries
export async function recordPacketRoute(route: InsertPacketRoute) {
  const db = await getDb();
  if (!db) return;
  await db.insert(packetRoutes).values(route);
}

export async function getRecentPacketRoutes(limit = 50) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(packetRoutes).orderBy(desc(packetRoutes.lastSeen)).limit(limit);
}
