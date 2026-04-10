import { useEffect, useState } from "react";
import { useLocation } from "wouter";
import { trpc } from "@/lib/trpc";
import { useAuth } from "@/_core/hooks/useAuth";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { AlertCircle, Wifi, Server, Network, Settings, Activity, LogOut } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

export default function ISPDashboard() {
  const [, setLocation] = useLocation();
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState("overview");
  const logoutMutation = trpc.auth.logout.useMutation();

  const handleLogout = async () => {
    await logoutMutation.mutateAsync();
    setLocation("/");
  };

  // Fetch service health
  const { data: health, isLoading: healthLoading } = trpc.isp.status.getHealth.useQuery(undefined, {
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Fetch DHCP leases
  const { data: leases } = trpc.isp.dhcp.getLeases.useQuery(undefined, {
    refetchInterval: 10000,
  });

  // Fetch DNS records
  const { data: dnsRecords } = trpc.isp.dns.getRecords.useQuery(undefined, {
    refetchInterval: 10000,
  });

  // Fetch connected clients
  const { data: clients } = trpc.isp.network.getClients.useQuery(undefined, {
    refetchInterval: 5000,
  });

  // Fetch system logs
  const { data: logs } = trpc.isp.logs.getRecent.useQuery({ limit: 50 }, {
    refetchInterval: 5000,
  });

  // Fetch gateway config
  const { data: gatewayConfig } = trpc.isp.gateway.getConfig.useQuery();

  const updateGatewayConfig = trpc.isp.gateway.updateConfig.useMutation();
  const addDNSRecord = trpc.isp.dns.addRecord.useMutation();
  const allocateIP = trpc.isp.dhcp.allocateIP.useMutation();

  const [newDomain, setNewDomain] = useState("");
  const [newIP, setNewIP] = useState("");
  const [newMac, setNewMac] = useState("");

  const handleAddDNS = async () => {
    if (newDomain && newIP) {
      await addDNSRecord.mutateAsync({
        domain: newDomain,
        ipAddress: newIP,
        ttl: 3600,
      });
      setNewDomain("");
      setNewIP("");
    }
  };

  const handleAllocateIP = async () => {
    if (newMac) {
      await allocateIP.mutateAsync({
        macAddress: newMac,
        leaseHours: 24,
      });
      setNewMac("");
    }
  };

  const getStatusColor = (enabled: boolean) => enabled ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800";

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">MeshISP Dashboard</h1>
              <p className="text-slate-300">Personal Internet Service Provider Management</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="text-sm text-slate-300">Logged in as</p>
                <p className="text-white font-semibold">{user?.name || user?.email}</p>
              </div>
              <Button
                onClick={handleLogout}
                variant="outline"
                className="flex items-center gap-2 bg-slate-700 border-slate-600 text-white hover:bg-slate-600"
              >
                <LogOut className="w-4 h-4" />
                Logout
              </Button>
            </div>
          </div>
        </div>

        {/* Service Status Overview */}
        {!healthLoading && health && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {/* DHCP Status */}
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Server className="w-5 h-5 text-blue-400" />
                  DHCP Server
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <Badge className={getStatusColor(health.dhcp.enabled)}>
                    {health.dhcp.enabled ? "Active" : "Inactive"}
                  </Badge>
                  <p className="text-sm text-slate-300">Active Leases: <span className="font-bold text-white">{health.dhcp.activeLeases}</span></p>
                  <p className="text-xs text-slate-400">Pool: {health.dhcp.poolStart} - {health.dhcp.poolEnd}</p>
                </div>
              </CardContent>
            </Card>

            {/* DNS Status */}
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Network className="w-5 h-5 text-purple-400" />
                  DNS Resolver
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <Badge className={getStatusColor(health.dns.enabled)}>
                    {health.dns.enabled ? "Active" : "Inactive"}
                  </Badge>
                  <p className="text-sm text-slate-300">Records: <span className="font-bold text-white">{health.dns.recordCount}</span></p>
                  <p className="text-xs text-slate-400">Upstream: {health.dns.upstreamServer}</p>
                </div>
              </CardContent>
            </Card>

            {/* Network Status */}
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Activity className="w-5 h-5 text-green-400" />
                  Network
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <Badge className="bg-green-100 text-green-800">Online</Badge>
                  <p className="text-sm text-slate-300">Clients: <span className="font-bold text-white">{health.network.onlineClients}</span></p>
                  <p className="text-xs text-slate-400">Gateway: {health.network.gatewayIp}</p>
                </div>
              </CardContent>
            </Card>

            {/* NAT Status */}
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Settings className="w-5 h-5 text-orange-400" />
                  NAT/Gateway
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <Badge className={getStatusColor(health.nat.enabled)}>
                    {health.nat.enabled ? "Enabled" : "Disabled"}
                  </Badge>
                  <p className="text-sm text-slate-300">External IP: <span className="font-bold text-white">{health.nat.externalIp || "N/A"}</span></p>
                  <p className="text-xs text-slate-400">Subnet: {health.network.subnet}</p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Main Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-5 bg-slate-800 border-slate-700">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="dhcp">DHCP</TabsTrigger>
            <TabsTrigger value="dns">DNS</TabsTrigger>
            <TabsTrigger value="clients">Clients</TabsTrigger>
            <TabsTrigger value="logs">Logs</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">System Overview</CardTitle>
                <CardDescription>Current ISP system status and configuration</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {gatewayConfig && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-slate-400">Gateway IP</p>
                      <p className="text-lg font-bold text-white">{gatewayConfig.gatewayIp}</p>
                    </div>
                    <div>
                      <p className="text-sm text-slate-400">Internal Subnet</p>
                      <p className="text-lg font-bold text-white">{gatewayConfig.internalSubnet}</p>
                    </div>
                    <div>
                      <p className="text-sm text-slate-400">DNS Server</p>
                      <p className="text-lg font-bold text-white">{gatewayConfig.dnsServer}</p>
                    </div>
                    <div>
                      <p className="text-sm text-slate-400">External IP</p>
                      <p className="text-lg font-bold text-white">{gatewayConfig.externalIp || "Not configured"}</p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* DHCP Tab */}
          <TabsContent value="dhcp" className="space-y-4">
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">DHCP Lease Management</CardTitle>
                <CardDescription>Allocate and manage IP addresses</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Input
                    placeholder="MAC Address (e.g., 00:11:22:33:44:55)"
                    value={newMac}
                    onChange={(e) => setNewMac(e.target.value)}
                    className="bg-slate-700 border-slate-600 text-white"
                  />
                  <Button onClick={handleAllocateIP} disabled={allocateIP.isPending}>
                    Allocate IP
                  </Button>
                </div>

                {leases && leases.length > 0 ? (
                  <div className="space-y-2">
                    <h3 className="text-sm font-semibold text-white">Active Leases</h3>
                    {leases.map((lease) => (
                      <div key={lease.id} className="bg-slate-700 p-3 rounded-lg flex justify-between items-center">
                        <div>
                          <p className="text-sm font-mono text-white">{lease.macAddress}</p>
                          <p className="text-xs text-slate-400">{lease.hostname || "Unknown"}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-bold text-blue-400">{lease.ipAddress}</p>
                          <p className="text-xs text-slate-400">
                            Expires: {new Date(lease.leaseEndTime).toLocaleString()}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-slate-400 text-sm">No active leases</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* DNS Tab */}
          <TabsContent value="dns" className="space-y-4">
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">DNS Records (.mesh domain)</CardTitle>
                <CardDescription>Manage local domain name resolution</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Input
                    placeholder="Domain (e.g., dashboard.mesh)"
                    value={newDomain}
                    onChange={(e) => setNewDomain(e.target.value)}
                    className="bg-slate-700 border-slate-600 text-white"
                  />
                  <Input
                    placeholder="IP Address"
                    value={newIP}
                    onChange={(e) => setNewIP(e.target.value)}
                    className="bg-slate-700 border-slate-600 text-white"
                  />
                  <Button onClick={handleAddDNS} disabled={addDNSRecord.isPending}>
                    Add Record
                  </Button>
                </div>

                {dnsRecords && dnsRecords.length > 0 ? (
                  <div className="space-y-2">
                    <h3 className="text-sm font-semibold text-white">DNS Records</h3>
                    {dnsRecords.map((record) => (
                      <div key={record.id} className="bg-slate-700 p-3 rounded-lg flex justify-between items-center">
                        <div>
                          <p className="text-sm font-semibold text-white">{record.domain}</p>
                          <p className="text-xs text-slate-400">{record.description || "No description"}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-bold text-purple-400">{record.ipAddress}</p>
                          <p className="text-xs text-slate-400">TTL: {record.ttl}s</p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-slate-400 text-sm">No DNS records configured</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Clients Tab */}
          <TabsContent value="clients" className="space-y-4">
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">Connected Clients</CardTitle>
                <CardDescription>Devices currently connected to the mesh network</CardDescription>
              </CardHeader>
              <CardContent>
                {clients && clients.length > 0 ? (
                  <div className="space-y-2">
                    {clients.map((client) => (
                      <div key={client.id} className="bg-slate-700 p-4 rounded-lg">
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <p className="text-sm font-semibold text-white">{client.hostname || "Unknown Device"}</p>
                            <p className="text-xs font-mono text-slate-400">{client.macAddress}</p>
                          </div>
                          <Badge className="bg-green-100 text-green-800">Online</Badge>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                          <div>
                            <p className="text-slate-400">IP Address</p>
                            <p className="font-bold text-blue-400">{client.ipAddress}</p>
                          </div>
                          <div>
                            <p className="text-slate-400">Signal Strength</p>
                            <p className="font-bold text-white">{client.signalStrength ? `${client.signalStrength}%` : "N/A"}</p>
                          </div>
                          <div>
                            <p className="text-slate-400">Last Seen</p>
                            <p className="font-bold text-white">{new Date(client.lastSeen).toLocaleTimeString()}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-slate-400 text-sm">No clients connected</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Logs Tab */}
          <TabsContent value="logs" className="space-y-4">
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">System Logs</CardTitle>
                <CardDescription>Real-time events from DHCP, DNS, WiFi, and NAT services</CardDescription>
              </CardHeader>
              <CardContent>
                {logs && logs.length > 0 ? (
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {logs.map((log) => (
                      <div key={log.id} className="bg-slate-700 p-3 rounded-lg text-xs font-mono">
                        <div className="flex justify-between items-start mb-1">
                          <span className="text-slate-400">{new Date(log.timestamp).toLocaleTimeString()}</span>
                          <Badge variant={log.level === "ERROR" ? "destructive" : "secondary"}>
                            {log.service}
                          </Badge>
                        </div>
                        <p className="text-white">{log.message}</p>
                        {log.details && <p className="text-slate-500 mt-1">{log.details}</p>}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-slate-400 text-sm">No logs available</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
