# MeshISP Dashboard - Project TODO

## Core Features

### Phase 1: Project Setup
- [x] Initialize web project with db, server, and user features
- [x] Create database schema for ISP system
- [x] Set up environment variables and configuration

### Phase 2: Database Schema
- [x] DHCP leases table (MAC, IP, lease time, renewal)
- [x] DNS records table (domain, IP, TTL, type)
- [x] Connected clients table (MAC, hostname, IP, signal strength)
- [x] System logs table (timestamp, service, level, message)
- [x] Gateway config table (external IP, subnet, NAT rules)
- [x] Network interfaces table (interface name, IP, status)

### Phase 3: Backend Services

#### DHCP Server
- [x] IP pool management (10.99.1.100 - 10.99.1.250)
- [x] Lease allocation and tracking
- [x] Lease renewal logic
- [x] Lease release handling
- [x] Persistent lease storage
- [x] DHCP request/response handling

#### DNS Resolver
- [x] Local .mesh domain resolution
- [x] DNS record management API
- [x] Upstream public DNS forwarding (8.8.8.8)
- [x] DNS query logging
- [ ] DNS cache implementation

#### WiFi Hotspot Controller
- [ ] Windows netsh wlan integration
- [ ] Hotspot creation and teardown
- [ ] Connected client detection
- [ ] MAC address tracking
- [ ] Signal strength monitoring

#### NAT & Routing Engine
- [ ] Internal to external traffic translation
- [ ] Packet forwarding logic
- [ ] Route table management
- [ ] Port forwarding configuration
- [ ] Traffic statistics collection

### Phase 4: Frontend Dashboard

#### Service Status
- [x] DHCP service health indicator
- [x] DNS service health indicator
- [x] WiFi hotspot status
- [x] Gateway connectivity status
- [x] Real-time uptime tracking

#### Network Monitoring
- [x] Connected clients list with MAC addresses
- [x] Assigned IP addresses display
- [ ] Signal strength visualization
- [ ] Bandwidth usage per client
- [x] Active connections count

#### DHCP Management
- [x] View active leases
- [x] Manual IP assignment
- [ ] Lease renewal/release controls
- [ ] IP pool configuration
- [ ] Lease history

#### DNS Management
- [x] Add/edit/delete .mesh domain records
- [x] View DNS query logs
- [ ] Configure upstream DNS servers
- [ ] DNS cache statistics

#### Gateway Configuration
- [x] External internet connectivity settings
- [ ] NAT rule management
- [ ] Port forwarding configuration
- [x] Subnet and gateway IP settings

#### Network Diagnostics
- [ ] Ping tool with target IP/hostname
- [ ] Traceroute visualization
- [ ] Bandwidth test between nodes
- [ ] Packet loss statistics
- [ ] Latency measurements

#### System Logs Viewer
- [x] Real-time log streaming
- [ ] Filter by service (DHCP, DNS, WiFi, NAT)
- [ ] Filter by log level (INFO, WARN, ERROR)
- [ ] Search and pagination
- [ ] Export logs to file

#### Packet Routing Visualization
- [ ] Real-time traffic flow diagram
- [ ] Node connectivity graph
- [ ] Packet throughput visualization
- [ ] Route path highlighting
- [ ] Traffic statistics overlay

### Phase 5: Integration & Testing
- [ ] Backend service integration tests
- [ ] Frontend component tests
- [ ] End-to-end workflow tests
- [ ] Performance testing
- [ ] Error handling and recovery

### Phase 6: Deployment & Documentation
- [ ] Setup documentation
- [ ] User guide
- [ ] API documentation
- [ ] Troubleshooting guide
- [ ] Final checkpoint

## Technical Stack
- **Frontend**: React 19 + Tailwind 4 + shadcn/ui
- **Backend**: Express 4 + tRPC 11
- **Database**: MySQL/TiDB
- **Services**: Python (DHCP, DNS, NAT, WiFi) + Node.js (Dashboard)
- **Real-time**: WebSockets for log streaming and packet visualization

## Notes
- This is a personal ISP system for AI assistant connectivity while traveling
- All services run on a single Windows machine (HP EliteDesk G4 with TP-Link WiFi 6 adapter)
- No commercial use or external service dependencies
- Focus on reliability and ease of management
