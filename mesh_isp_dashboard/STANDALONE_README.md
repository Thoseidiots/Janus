# MeshISP - Standalone Personal Internet Service Provider

A fully self-contained, portable ISP management system for personal use. No external API keys, no cloud dependencies, no OAuth required.

## Features

### Core ISP Services
- **DHCP Server**: Automatic IP address allocation (10.99.1.100-10.99.1.250), lease management, and renewal tracking
- **DNS Resolver**: Local .mesh domain resolution with upstream forwarding to public DNS
- **Network Monitoring**: Real-time client tracking, bandwidth monitoring, and packet routing visualization
- **Gateway Configuration**: External connectivity settings, NAT management, and service toggles
- **System Logs**: Real-time event streaming with service filtering and detailed diagnostics

### Authentication
- **Local Authentication**: Simple email/password login system (no OAuth, no external services)
- **User Management**: Create accounts, manage roles, and track login history
- **Session Management**: Secure cookie-based sessions

## Quick Start

### Prerequisites
- Node.js 18+ and pnpm
- MySQL 8.0+ or compatible database (TiDB, MariaDB)
- Windows (for WiFi hotspot features) or Linux/Mac (for core ISP functions)

### Installation

1. **Extract the project**
   ```bash
   unzip mesh-isp-dashboard.zip
   cd mesh-isp-dashboard
   ```

2. **Install dependencies**
   ```bash
   pnpm install
   ```

3. **Set up your database**
   
   Create a MySQL database:
   ```sql
   CREATE DATABASE mesh_isp;
   ```

   Create a `.env.local` file in the project root:
   ```env
   DATABASE_URL=mysql://username:password@localhost:3306/mesh_isp
   JWT_SECRET=your-secret-key-here-change-this
   ```

4. **Run database migrations**
   ```bash
   pnpm drizzle-kit migrate
   ```

5. **Start the development server**
   ```bash
   pnpm dev
   ```

6. **Access the application**
   - Open http://localhost:3000 in your browser
   - Create a new account or log in
   - Access the ISP Dashboard at http://localhost:3000/isp

## Project Structure

```
mesh-isp-dashboard/
├── client/                 # React frontend
│   └── src/
│       ├── pages/         # Page components (Home, Login, ISPDashboard)
│       ├── components/    # Reusable UI components
│       └── lib/           # tRPC client setup
├── server/                # Node.js backend
│   ├── routers/
│   │   ├── isp.ts        # ISP service endpoints
│   │   └── auth.ts       # Local authentication
│   ├── auth.ts           # Authentication logic
│   ├── db.ts             # Database queries
│   └── _core/            # Framework internals
├── drizzle/              # Database schema and migrations
│   ├── schema.ts         # Table definitions
│   └── migrations/       # SQL migration files
├── package.json          # Dependencies
└── README.md             # This file
```

## API Endpoints

All endpoints are protected by local authentication. Access via tRPC client.

### DHCP Management
- `isp.dhcp.getLeases()` - Get all active DHCP leases
- `isp.dhcp.allocateIP()` - Allocate a new IP to a device
- `isp.dhcp.getLease()` - Get a specific lease by MAC address
- `isp.dhcp.releaseIP()` - Release an IP address

### DNS Management
- `isp.dns.getRecords()` - Get all DNS records
- `isp.dns.addRecord()` - Add a new .mesh domain record
- `isp.dns.getRecord()` - Get a specific DNS record
- `isp.dns.deleteRecord()` - Delete a DNS record

### Network Monitoring
- `isp.network.getClients()` - Get connected clients
- `isp.network.updateClient()` - Update client information
- `isp.network.getPacketRoutes()` - Get packet routing data
- `isp.network.recordTraffic()` - Record traffic statistics

### Gateway Configuration
- `isp.gateway.getConfig()` - Get gateway settings
- `isp.gateway.updateConfig()` - Update gateway configuration

### System Logs
- `isp.logs.getRecent()` - Get recent system logs
- `isp.logs.addLog()` - Add a system log entry

### Service Status
- `isp.status.getHealth()` - Get overall system health status

## Database Schema

### Users Table
- `id`: Primary key
- `openId`: Unique identifier (email for local auth)
- `email`: User email
- `name`: User display name
- `loginMethod`: Authentication method (local)
- `role`: User role (user/admin)
- `createdAt`, `updatedAt`, `lastSignedIn`: Timestamps

### DHCP Leases Table
- `macAddress`: Device MAC address (unique)
- `ipAddress`: Assigned IP address
- `hostname`: Device hostname
- `leaseStartTime`, `leaseEndTime`: Lease duration
- `renewalTime`: When lease should be renewed
- `isActive`: Whether lease is active

### DNS Records Table
- `domain`: Domain name (unique)
- `recordType`: DNS record type (A, AAAA, etc.)
- `ipAddress`: IP address to resolve to
- `ttl`: Time to live in seconds
- `description`: Record description

### Connected Clients Table
- `macAddress`: Device MAC address (unique)
- `hostname`: Device hostname
- `ipAddress`: Current IP address
- `signalStrength`: WiFi signal strength (0-100)
- `isOnline`: Whether device is currently online
- `bandwidthUsage`: Current bandwidth usage
- `lastSeen`: Last activity timestamp

### System Logs Table
- `timestamp`: Log timestamp
- `service`: Service name (DHCP, DNS, WiFi, NAT, etc.)
- `level`: Log level (INFO, WARN, ERROR, DEBUG)
- `message`: Log message
- `details`: Additional details

### Gateway Config Table
- `externalIp`: External internet IP
- `internalSubnet`: Internal network subnet (10.99.1.0/24)
- `gatewayIp`: Gateway IP address (10.99.1.1)
- `dnsServer`: Upstream DNS server (8.8.8.8)
- `natEnabled`: Whether NAT is enabled
- `dhcpEnabled`: Whether DHCP is enabled
- `dnsEnabled`: Whether DNS is enabled

## Configuration

### Environment Variables

Create a `.env.local` file:

```env
# Database
DATABASE_URL=mysql://user:password@localhost:3306/mesh_isp

# Security
JWT_SECRET=your-secret-key-change-this-in-production

# Optional
NODE_ENV=development
PORT=3000
```

### Gateway Settings

Configure via the ISP Dashboard:
- **Gateway IP**: Default 10.99.1.1
- **Internal Subnet**: Default 10.99.1.0/24
- **DNS Server**: Default 8.8.8.8
- **External IP**: Your public internet IP (optional)

## Development

### Running Tests
```bash
pnpm test
```

### Type Checking
```bash
pnpm check
```

### Building for Production
```bash
pnpm build
```

### Starting Production Server
```bash
pnpm start
```

## Windows WiFi Hotspot Setup (Optional)

To enable WiFi hotspot features on Windows:

1. Install the Python dependencies (from the included `RealMeshISP` folder):
   ```bash
   pip install flask waitress dnslib scapy
   ```

2. Run the Windows WiFi controller:
   ```bash
   python isp_engine.py
   ```

3. The system will use `netsh wlan` commands to manage the hosted network

## Troubleshooting

### Database Connection Error
- Verify MySQL is running
- Check DATABASE_URL in .env.local
- Ensure database exists and user has permissions

### Port Already in Use
- Change PORT in .env.local
- Or kill the process using port 3000: `lsof -ti:3000 | xargs kill -9`

### Login Not Working
- Clear browser cookies
- Check that database migrations have run
- Verify user was created successfully

### ISP Dashboard Not Loading
- Ensure you're logged in (check /login page)
- Check browser console for errors
- Verify backend is running: `pnpm dev`

## Security Notes

⚠️ **Important**: This is a personal ISP system for local use. For production deployments:

1. **Change JWT_SECRET** to a strong random value
2. **Use HTTPS** in production
3. **Implement password hashing** (currently uses plaintext for demo)
4. **Restrict network access** to trusted devices only
5. **Regular backups** of your database
6. **Keep dependencies updated** with `pnpm update`

## License

This project is provided as-is for personal use.

## Support

This is a standalone system with no external dependencies. All functionality is self-contained:
- No API keys required
- No cloud services needed
- No external authentication
- Works completely offline (except for upstream DNS)

For issues or questions, refer to the code comments and database schema documentation above.

## Next Steps

1. **Configure your gateway** - Set external IP and DNS settings
2. **Add DNS records** - Create .mesh domain entries for your devices
3. **Monitor network** - View connected clients and system logs
4. **Manage leases** - Allocate and track DHCP leases
5. **Enable WiFi hotspot** - Use Windows WiFi controller for connectivity

---

**MeshISP** - Your personal, self-contained Internet Service Provider system.
