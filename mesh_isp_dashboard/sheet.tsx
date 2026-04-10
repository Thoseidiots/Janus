import { useAuth } from "@/_core/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "wouter";
import { Wifi, Server, Network, Settings, Zap, Shield } from "lucide-react";

export default function Home() {
  const { user, isAuthenticated } = useAuth();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-700 bg-slate-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Wifi className="w-8 h-8 text-blue-400" />
            <h1 className="text-2xl font-bold text-white">MeshISP</h1>
          </div>
          <div>
            {isAuthenticated ? (
              <Link href="/isp">
                <Button className="bg-blue-600 hover:bg-blue-700">Dashboard</Button>
              </Link>
            ) : (
              <Link href="/login">
                <Button className="bg-blue-600 hover:bg-blue-700">Sign In</Button>
              </Link>
            )}
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-6 py-20">
        <div className="text-center mb-16">
          <h2 className="text-5xl font-bold text-white mb-4">Personal Internet Service Provider</h2>
          <p className="text-xl text-slate-300 mb-8">
            A real ISP system for personal use. Manage DHCP, DNS, WiFi, and network routing from anywhere.
          </p>
          {isAuthenticated ? (
            <Link href="/isp">
              <Button size="lg" className="bg-blue-600 hover:bg-blue-700 text-lg px-8">
                Go to Dashboard
              </Button>
            </Link>
          ) : (
            <Link href="/login">
              <Button size="lg" className="bg-blue-600 hover:bg-blue-700 text-lg px-8">
                Get Started
              </Button>
            </Link>
          )}
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* DHCP */}
          <Card className="bg-slate-800 border-slate-700 hover:border-blue-500 transition">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Server className="w-5 h-5 text-blue-400" />
                DHCP Server
              </CardTitle>
              <CardDescription>IP address allocation and lease management</CardDescription>
            </CardHeader>
            <CardContent className="text-slate-300">
              <ul className="space-y-2 text-sm">
                <li>• Automatic IP pool management</li>
                <li>• Lease tracking and renewal</li>
                <li>• MAC address binding</li>
                <li>• Real-time lease status</li>
              </ul>
            </CardContent>
          </Card>

          {/* DNS */}
          <Card className="bg-slate-800 border-slate-700 hover:border-purple-500 transition">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Network className="w-5 h-5 text-purple-400" />
                DNS Resolver
              </CardTitle>
              <CardDescription>Local .mesh domain resolution</CardDescription>
            </CardHeader>
            <CardContent className="text-slate-300">
              <ul className="space-y-2 text-sm">
                <li>• .mesh domain support</li>
                <li>• Upstream DNS forwarding</li>
                <li>• Custom domain records</li>
                <li>• Query logging</li>
              </ul>
            </CardContent>
          </Card>

          {/* WiFi Hotspot */}
          <Card className="bg-slate-800 border-slate-700 hover:border-green-500 transition">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Wifi className="w-5 h-5 text-green-400" />
                WiFi Hotspot
              </CardTitle>
              <CardDescription>Windows hosted network control</CardDescription>
            </CardHeader>
            <CardContent className="text-slate-300">
              <ul className="space-y-2 text-sm">
                <li>• Hotspot creation & management</li>
                <li>• Connected client tracking</li>
                <li>• Signal strength monitoring</li>
                <li>• Real-time status updates</li>
              </ul>
            </CardContent>
          </Card>

          {/* NAT & Routing */}
          <Card className="bg-slate-800 border-slate-700 hover:border-orange-500 transition">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Zap className="w-5 h-5 text-orange-400" />
                NAT & Routing
              </CardTitle>
              <CardDescription>Network address translation and traffic routing</CardDescription>
            </CardHeader>
            <CardContent className="text-slate-300">
              <ul className="space-y-2 text-sm">
                <li>• Internal to external translation</li>
                <li>• Packet forwarding</li>
                <li>• Route table management</li>
                <li>• Traffic statistics</li>
              </ul>
            </CardContent>
          </Card>

          {/* Network Monitoring */}
          <Card className="bg-slate-800 border-slate-700 hover:border-cyan-500 transition">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Shield className="w-5 h-5 text-cyan-400" />
                Network Monitoring
              </CardTitle>
              <CardDescription>Real-time network visibility</CardDescription>
            </CardHeader>
            <CardContent className="text-slate-300">
              <ul className="space-y-2 text-sm">
                <li>• Connected client list</li>
                <li>• Bandwidth usage tracking</li>
                <li>• Packet routing visualization</li>
                <li>• Performance metrics</li>
              </ul>
            </CardContent>
          </Card>

          {/* System Logs */}
          <Card className="bg-slate-800 border-slate-700 hover:border-red-500 transition">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Settings className="w-5 h-5 text-red-400" />
                System Logs
              </CardTitle>
              <CardDescription>Service event tracking and diagnostics</CardDescription>
            </CardHeader>
            <CardContent className="text-slate-300">
              <ul className="space-y-2 text-sm">
                <li>• Real-time log streaming</li>
                <li>• Service-based filtering</li>
                <li>• Error tracking</li>
                <li>• Event history</li>
              </ul>
            </CardContent>
          </Card>
        </div>

        {/* Use Case */}
        <section className="mt-20 bg-slate-800 border border-slate-700 rounded-lg p-8">
          <h3 className="text-2xl font-bold text-white mb-4">Use Case: AI Assistant Connectivity</h3>
          <p className="text-slate-300 mb-4">
            This ISP system is designed for personal use to enable communication with your AI assistant when traveling outside your home state. 
            By running this system on a dedicated machine with a WiFi adapter, you create a private mesh network that:
          </p>
          <ul className="space-y-2 text-slate-300 list-disc list-inside">
            <li>Provides reliable local network connectivity</li>
            <li>Manages IP addresses automatically via DHCP</li>
            <li>Resolves local domain names via DNS</li>
            <li>Routes traffic through NAT to external internet</li>
            <li>Monitors all network activity in real-time</li>
          </ul>
        </section>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-700 bg-slate-900/50 backdrop-blur mt-20">
        <div className="max-w-7xl mx-auto px-6 py-8 text-center text-slate-400">
          <p>MeshISP - Personal Internet Service Provider System</p>
          <p className="text-sm mt-2">For personal use only. Not for commercial purposes.</p>
        </div>
      </footer>
    </div>
  );
}
