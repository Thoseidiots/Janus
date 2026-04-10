/**
 * Windows WiFi Hotspot Control Module
 * Manages Windows Hosted Network (WiFi hotspot) using netsh commands
 * 
 * Requires: Windows 7 or later with WiFi adapter
 * Admin privileges: Required for netsh wlan commands
 * 
 * Usage:
 *   const hotspot = new WindowsHotspotManager();
 *   await hotspot.start('MyNetwork', 'password123');
 *   await hotspot.stop();
 */

import { execSync, spawn } from 'child_process';
import { promisify } from 'util';
import { exec } from 'child_process';

const execAsync = promisify(exec);

export interface HotspotConfig {
  ssid: string;
  password: string;
  maxClients?: number;
}

export interface HotspotStatus {
  isRunning: boolean;
  ssid?: string;
  status?: string;
  connectedClients?: number;
  errorMessage?: string;
}

export class WindowsHotspotManager {
  private isRunning: boolean = false;
  private currentSSID: string = '';
  private currentPassword: string = '';

  /**
   * Check if Windows Hosted Network is available
   */
  async isAvailable(): Promise<boolean> {
    try {
      const { stdout } = await execAsync('netsh wlan show drivers');
      return stdout.includes('Hosted network supported') && stdout.includes('Yes');
    } catch (error) {
      console.error('[HOTSPOT] Error checking availability:', error);
      return false;
    }
  }

  /**
   * Start the WiFi hotspot
   * @param config - Hotspot configuration (SSID and password)
   */
  async start(config: HotspotConfig): Promise<{ success: boolean; message: string }> {
    try {
      // Validate input
      if (!config.ssid || config.ssid.length === 0) {
        throw new Error('SSID cannot be empty');
      }
      if (!config.password || config.password.length < 8) {
        throw new Error('Password must be at least 8 characters');
      }
      if (config.ssid.length > 32) {
        throw new Error('SSID cannot exceed 32 characters');
      }

      console.log(`[HOTSPOT] Starting hotspot with SSID: ${config.ssid}`);

      // Set the hosted network SSID and password
      const setCommand = `netsh wlan set hostednetwork mode=allow ssid="${config.ssid}" key="${config.password}"`;
      await execAsync(setCommand);

      // Start the hosted network
      const startCommand = 'netsh wlan start hostednetwork';
      await execAsync(startCommand);

      this.isRunning = true;
      this.currentSSID = config.ssid;
      this.currentPassword = config.password;

      console.log(`[HOTSPOT] Hotspot started successfully: ${config.ssid}`);
      return {
        success: true,
        message: `WiFi hotspot "${config.ssid}" started successfully`,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('[HOTSPOT] Error starting hotspot:', errorMessage);
      this.isRunning = false;
      return {
        success: false,
        message: `Failed to start hotspot: ${errorMessage}`,
      };
    }
  }

  /**
   * Stop the WiFi hotspot
   */
  async stop(): Promise<{ success: boolean; message: string }> {
    try {
      console.log('[HOTSPOT] Stopping hotspot...');

      const stopCommand = 'netsh wlan stop hostednetwork';
      await execAsync(stopCommand);

      this.isRunning = false;
      this.currentSSID = '';
      this.currentPassword = '';

      console.log('[HOTSPOT] Hotspot stopped successfully');
      return {
        success: true,
        message: 'WiFi hotspot stopped successfully',
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('[HOTSPOT] Error stopping hotspot:', errorMessage);
      return {
        success: false,
        message: `Failed to stop hotspot: ${errorMessage}`,
      };
    }
  }

  /**
   * Get current hotspot status
   */
  async getStatus(): Promise<HotspotStatus> {
    try {
      const { stdout } = await execAsync('netsh wlan show hostednetwork');

      const status: HotspotStatus = {
        isRunning: this.isRunning,
      };

      // Parse SSID
      const ssidMatch = stdout.match(/SSID\s*:\s*(.+)/);
      if (ssidMatch) {
        status.ssid = ssidMatch[1].trim();
      }

      // Parse status
      const statusMatch = stdout.match(/Status\s*:\s*(.+)/);
      if (statusMatch) {
        status.status = statusMatch[1].trim();
        status.isRunning = statusMatch[1].toLowerCase().includes('started');
      }

      // Parse connected clients
      const clientsMatch = stdout.match(/Number of clients\s*:\s*(\d+)/);
      if (clientsMatch) {
        status.connectedClients = parseInt(clientsMatch[1], 10);
      }

      return status;
    } catch (error) {
      console.error('[HOTSPOT] Error getting status:', error);
      return {
        isRunning: false,
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Get list of connected clients
   */
  async getConnectedClients(): Promise<Array<{ macAddress: string; ipAddress: string }>> {
    try {
      const { stdout } = await execAsync('netsh wlan show hostednetwork');

      const clients: Array<{ macAddress: string; ipAddress: string }> = [];

      // Parse client information from netsh output
      // Format: MAC Address : XX:XX:XX:XX:XX:XX
      const clientMatches = stdout.matchAll(/MAC Address\s*:\s*([0-9A-Fa-f:]+)/g);

      for (const match of clientMatches) {
        clients.push({
          macAddress: match[1],
          ipAddress: '', // netsh doesn't directly provide IP, would need additional lookup
        });
      }

      return clients;
    } catch (error) {
      console.error('[HOTSPOT] Error getting connected clients:', error);
      return [];
    }
  }

  /**
   * Refresh the hotspot (stop and start)
   */
  async refresh(config: HotspotConfig): Promise<{ success: boolean; message: string }> {
    try {
      console.log('[HOTSPOT] Refreshing hotspot...');

      // Stop current hotspot
      await this.stop();

      // Wait a moment for cleanup
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Start new hotspot
      return await this.start(config);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('[HOTSPOT] Error refreshing hotspot:', errorMessage);
      return {
        success: false,
        message: `Failed to refresh hotspot: ${errorMessage}`,
      };
    }
  }

  /**
   * Get current SSID
   */
  getCurrentSSID(): string {
    return this.currentSSID;
  }

  /**
   * Check if hotspot is running
   */
  isHotspotRunning(): boolean {
    return this.isRunning;
  }

  /**
   * Set WiFi adapter to bridge mode (advanced)
   * This allows the hotspot to share internet from an existing connection
   */
  async setBridgeMode(adapterName: string): Promise<{ success: boolean; message: string }> {
    try {
      console.log(`[HOTSPOT] Setting bridge mode for adapter: ${adapterName}`);

      // This is a simplified approach - full bridging requires netsh interface commands
      const bridgeCommand = `netsh interface set interface "${adapterName}" forwarding=enabled`;
      await execAsync(bridgeCommand);

      return {
        success: true,
        message: `Bridge mode enabled for adapter: ${adapterName}`,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('[HOTSPOT] Error setting bridge mode:', errorMessage);
      return {
        success: false,
        message: `Failed to set bridge mode: ${errorMessage}`,
      };
    }
  }

  /**
   * Get available WiFi adapters
   */
  async getWiFiAdapters(): Promise<string[]> {
    try {
      const { stdout } = await execAsync('netsh wlan show interfaces');

      const adapters: string[] = [];
      const interfaceMatches = stdout.matchAll(/Interface Name\s*:\s*(.+)/g);

      for (const match of interfaceMatches) {
        adapters.push(match[1].trim());
      }

      return adapters;
    } catch (error) {
      console.error('[HOTSPOT] Error getting WiFi adapters:', error);
      return [];
    }
  }
}

/**
 * Singleton instance of the hotspot manager
 */
let hotspotInstance: WindowsHotspotManager | null = null;

/**
 * Get or create the hotspot manager instance
 */
export function getHotspotManager(): WindowsHotspotManager {
  if (!hotspotInstance) {
    hotspotInstance = new WindowsHotspotManager();
  }
  return hotspotInstance;
}
