/**
 * Bandwidth Throttling and QoS Module
 * Implements per-client bandwidth limiting and quality of service rules
 * 
 * Features:
 *   - Per-client download/upload speed limits
 *   - Token bucket algorithm for fair rate limiting
 *   - Priority-based QoS (high/normal/low)
 *   - Traffic shaping and burst allowance
 *   - Real-time bandwidth monitoring
 * 
 * Usage:
 *   const throttle = new BandwidthThrottler();
 *   throttle.setClientLimit('00:11:22:33:44:55', { download: 10, upload: 5 }); // Mbps
 *   const allowed = await throttle.checkLimit('00:11:22:33:44:55', 1024); // bytes
 */

export interface BandwidthLimit {
  download: number; // Mbps
  upload: number; // Mbps
  burst?: number; // Burst allowance in MB
  priority?: 'high' | 'normal' | 'low'; // QoS priority
}

export interface ClientBandwidth {
  macAddress: string;
  downloadUsed: number; // bytes
  uploadUsed: number; // bytes
  downloadLimit: number; // bps
  uploadLimit: number; // bps
  priority: 'high' | 'normal' | 'low';
  tokens: number; // Token bucket tokens
  lastUpdate: number; // Timestamp
}

export interface ThrottleStatus {
  macAddress: string;
  downloadMbps: number;
  uploadMbps: number;
  downloadUsedPercent: number;
  uploadUsedPercent: number;
  priority: string;
  isThrottled: boolean;
}

/**
 * Bandwidth Throttler - Implements token bucket algorithm for rate limiting
 */
export class BandwidthThrottler {
  private clients: Map<string, ClientBandwidth> = new Map();
  private defaultLimit: BandwidthLimit = {
    download: 100, // 100 Mbps default
    upload: 20, // 20 Mbps default
    burst: 10, // 10 MB burst
    priority: 'normal',
  };

  // Token bucket parameters
  private readonly REFILL_INTERVAL = 100; // ms - how often to refill tokens
  private readonly BYTES_PER_SECOND_TO_TOKENS = 1; // 1 byte = 1 token

  constructor() {
    // Periodically refill tokens for all clients
    setInterval(() => this.refillTokens(), this.REFILL_INTERVAL);
  }

  /**
   * Set bandwidth limit for a client
   * @param macAddress - Client MAC address
   * @param limit - Bandwidth limit configuration
   */
  setClientLimit(macAddress: string, limit: BandwidthLimit): void {
    const normalizedMac = this.normalizeMacAddress(macAddress);

    // Convert Mbps to bytes per second
    const downloadBps = limit.download * 1_000_000 / 8;
    const uploadBps = limit.upload * 1_000_000 / 8;

    const existing = this.clients.get(normalizedMac);

    this.clients.set(normalizedMac, {
      macAddress: normalizedMac,
      downloadUsed: existing?.downloadUsed || 0,
      uploadUsed: existing?.uploadUsed || 0,
      downloadLimit: downloadBps,
      uploadLimit: uploadBps,
      priority: limit.priority || 'normal',
      tokens: (downloadBps + uploadBps) * (limit.burst || 10), // Initial tokens for burst
      lastUpdate: Date.now(),
    });

    console.log(
      `[QoS] Set limit for ${normalizedMac}: ${limit.download}Mbps down, ${limit.upload}Mbps up`
    );
  }

  /**
   * Remove bandwidth limit for a client (use default)
   */
  removeClientLimit(macAddress: string): void {
    const normalizedMac = this.normalizeMacAddress(macAddress);
    this.clients.delete(normalizedMac);
    console.log(`[QoS] Removed limit for ${normalizedMac}`);
  }

  /**
   * Check if data transfer is allowed (token bucket check)
   * @param macAddress - Client MAC address
   * @param bytes - Number of bytes to transfer
   * @param direction - 'download' or 'upload'
   * @returns boolean - True if transfer is allowed
   */
  async checkLimit(
    macAddress: string,
    bytes: number,
    direction: 'download' | 'upload' = 'download'
  ): Promise<boolean> {
    const normalizedMac = this.normalizeMacAddress(macAddress);
    let client = this.clients.get(normalizedMac);

    // Create client with default limits if doesn't exist
    if (!client) {
      this.setClientLimit(normalizedMac, this.defaultLimit);
      client = this.clients.get(normalizedMac)!;
    }

    // Check if we have enough tokens
    if (client.tokens >= bytes) {
      // Deduct tokens
      client.tokens -= bytes;

      // Update usage
      if (direction === 'download') {
        client.downloadUsed += bytes;
      } else {
        client.uploadUsed += bytes;
      }

      return true;
    }

    // Not enough tokens - request is throttled
    console.warn(
      `[QoS] Throttling ${normalizedMac}: ${bytes} bytes, only ${Math.floor(client.tokens)} tokens available`
    );
    return false;
  }

  /**
   * Get throttle status for a client
   */
  getClientStatus(macAddress: string): ThrottleStatus | null {
    const normalizedMac = this.normalizeMacAddress(macAddress);
    const client = this.clients.get(normalizedMac);

    if (!client) {
      return null;
    }

    return {
      macAddress: normalizedMac,
      downloadMbps: (client.downloadLimit * 8) / 1_000_000,
      uploadMbps: (client.uploadLimit * 8) / 1_000_000,
      downloadUsedPercent: (client.downloadUsed / (client.downloadLimit * 3600)) * 100,
      uploadUsedPercent: (client.uploadUsed / (client.uploadLimit * 3600)) * 100,
      priority: client.priority,
      isThrottled: client.tokens < client.downloadLimit + client.uploadLimit,
    };
  }

  /**
   * Get all client statuses
   */
  getAllClientStatuses(): ThrottleStatus[] {
    const statuses: ThrottleStatus[] = [];

    for (const [, client] of this.clients) {
      statuses.push({
        macAddress: client.macAddress,
        downloadMbps: (client.downloadLimit * 8) / 1_000_000,
        uploadMbps: (client.uploadLimit * 8) / 1_000_000,
        downloadUsedPercent: (client.downloadUsed / (client.downloadLimit * 3600)) * 100,
        uploadUsedPercent: (client.uploadUsed / (client.uploadLimit * 3600)) * 100,
        priority: client.priority,
        isThrottled: client.tokens < client.downloadLimit + client.uploadLimit,
      });
    }

    return statuses;
  }

  /**
   * Reset usage statistics for a client
   */
  resetClientUsage(macAddress: string): void {
    const normalizedMac = this.normalizeMacAddress(macAddress);
    const client = this.clients.get(normalizedMac);

    if (client) {
      client.downloadUsed = 0;
      client.uploadUsed = 0;
      console.log(`[QoS] Reset usage for ${normalizedMac}`);
    }
  }

  /**
   * Reset all usage statistics
   */
  resetAllUsage(): void {
    for (const [, client] of this.clients) {
      client.downloadUsed = 0;
      client.uploadUsed = 0;
    }
    console.log('[QoS] Reset usage for all clients');
  }

  /**
   * Get current bandwidth usage summary
   */
  getBandwidthSummary(): {
    totalClients: number;
    throttledClients: number;
    totalDownloadMbps: number;
    totalUploadMbps: number;
  } {
    let throttledCount = 0;
    let totalDownloadMbps = 0;
    let totalUploadMbps = 0;

    for (const [, client] of this.clients) {
      totalDownloadMbps += (client.downloadLimit * 8) / 1_000_000;
      totalUploadMbps += (client.uploadLimit * 8) / 1_000_000;

      if (client.tokens < client.downloadLimit + client.uploadLimit) {
        throttledCount++;
      }
    }

    return {
      totalClients: this.clients.size,
      throttledClients: throttledCount,
      totalDownloadMbps,
      totalUploadMbps,
    };
  }

  /**
   * Refill tokens for all clients (called periodically)
   * @private
   */
  private refillTokens(): void {
    const now = Date.now();
    const timeSinceLastRefill = this.REFILL_INTERVAL / 1000; // seconds

    for (const [, client] of this.clients) {
      // Calculate tokens to add based on time elapsed and bandwidth limit
      const tokensToAdd =
        (client.downloadLimit + client.uploadLimit) * timeSinceLastRefill * this.BYTES_PER_SECOND_TO_TOKENS;

      // Add tokens but don't exceed burst capacity
      const maxTokens = (client.downloadLimit + client.uploadLimit) * 10; // 10 second burst
      client.tokens = Math.min(client.tokens + tokensToAdd, maxTokens);

      client.lastUpdate = now;
    }
  }

  /**
   * Normalize MAC address to standard format (XX:XX:XX:XX:XX:XX)
   * @private
   */
  private normalizeMacAddress(mac: string): string {
    return mac.toUpperCase().replace(/-/g, ':');
  }
}

/**
 * Singleton instance of the bandwidth throttler
 */
let throttlerInstance: BandwidthThrottler | null = null;

/**
 * Get or create the bandwidth throttler instance
 */
export function getThrottler(): BandwidthThrottler {
  if (!throttlerInstance) {
    throttlerInstance = new BandwidthThrottler();
  }
  return throttlerInstance;
}
