# MeshISP Setup Instructions

## No External Dependencies Required

This is a fully standalone system. You don't need any API keys, OAuth, or cloud services.

## Step 1: Database Setup

### Option A: MySQL (Recommended)

1. **Install MySQL**
   - Windows: Download from https://dev.mysql.com/downloads/mysql/
   - Mac: `brew install mysql`
   - Linux: `sudo apt-get install mysql-server`

2. **Start MySQL**
   ```bash
   # Windows: MySQL should start automatically
   # Mac: brew services start mysql
   # Linux: sudo systemctl start mysql
   ```

3. **Create database and user**
   ```bash
   mysql -u root -p
   ```
   
   Then run:
   ```sql
   CREATE DATABASE mesh_isp;
   CREATE USER 'mesh_user'@'localhost' IDENTIFIED BY 'your_password_here';
   GRANT ALL PRIVILEGES ON mesh_isp.* TO 'mesh_user'@'localhost';
   FLUSH PRIVILEGES;
   EXIT;
   ```

### Option B: SQLite (Simpler, File-based)

SQLite requires no installation. Just update your DATABASE_URL:
```
DATABASE_URL=file:./mesh_isp.db
```

## Step 2: Project Setup

1. **Extract the ZIP file**
   ```bash
   unzip mesh-isp-dashboard.zip
   cd mesh-isp-dashboard
   ```

2. **Install Node.js dependencies**
   ```bash
   pnpm install
   ```
   
   (If you don't have pnpm: `npm install -g pnpm`)

3. **Create .env.local file** in the project root:
   
   For MySQL:
   ```
   DATABASE_URL=mysql://mesh_user:your_password_here@localhost:3306/mesh_isp
   JWT_SECRET=change-this-to-a-random-string-12345
   NODE_ENV=development
   PORT=3000
   ```
   
   For SQLite:
   ```
   DATABASE_URL=file:./mesh_isp.db
   JWT_SECRET=change-this-to-a-random-string-12345
   NODE_ENV=development
   PORT=3000
   ```

4. **Run database migrations**
   ```bash
   pnpm drizzle-kit migrate
   ```

## Step 3: Start the Application

1. **Start the development server**
   ```bash
   pnpm dev
   ```

2. **Open in browser**
   - Navigate to http://localhost:3000
   - You should see the MeshISP home page

## Step 4: Create Your Account

1. **Click "Get Started"** on the home page
2. **Register a new account**
   - Email: your@email.com
   - Name: Your Name
   - Password: your-password
3. **You'll be automatically logged in** and redirected to the dashboard

## Step 5: Configure Your ISP

### Access the Dashboard
- URL: http://localhost:3000/isp
- You should see:
  - Service status cards (DHCP, DNS, Network, NAT)
  - Tabs for Overview, DHCP, DNS, Clients, and Logs

### Add DNS Records
1. Click the **DNS** tab
2. Enter a domain (e.g., `dashboard.mesh`)
3. Enter an IP address (e.g., `10.99.1.1`)
4. Click **Add Record**

### Allocate DHCP Leases
1. Click the **DHCP** tab
2. Enter a MAC address (e.g., `00:11:22:33:44:55`)
3. Click **Allocate IP**
4. The system will assign an IP from the pool (10.99.1.100-10.99.1.250)

### Monitor Clients
1. Click the **Clients** tab
2. View all connected devices with:
   - MAC address
   - Hostname
   - IP address
   - Signal strength
   - Last seen time

### View System Logs
1. Click the **Logs** tab
2. See real-time events from all services
3. Logs include DHCP requests, DNS queries, and system events

## Step 6: Production Deployment

### For Local Network Use

1. **Change JWT_SECRET** to a random string:
   ```bash
   # Generate a random secret
   node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
   ```

2. **Update .env.local** with the new secret

3. **Build for production**
   ```bash
   pnpm build
   ```

4. **Start production server**
   ```bash
   pnpm start
   ```

### For External Access (Advanced)

1. **Use a reverse proxy** (nginx, Apache)
2. **Enable HTTPS** with Let's Encrypt
3. **Restrict access** with firewall rules
4. **Regular backups** of your database

## Troubleshooting

### "Cannot connect to database"
- Check DATABASE_URL in .env.local
- Verify MySQL is running: `mysql -u root -p`
- For SQLite, ensure write permissions in project directory

### "Port 3000 already in use"
- Change PORT in .env.local to 3001, 3002, etc.
- Or kill the process: `lsof -ti:3000 | xargs kill -9`

### "Migration failed"
- Delete the database and recreate it
- Run migrations again: `pnpm drizzle-kit migrate`

### "Login not working"
- Clear browser cookies (Ctrl+Shift+Delete)
- Check that migrations completed successfully
- Verify user was created in database

### "Dashboard shows 'Loading...'"
- Wait 5-10 seconds for initial data load
- Check browser console for errors (F12)
- Verify backend is running

## Next Steps

1. **Configure gateway settings** - Set external IP and DNS server
2. **Add more DNS records** - Create entries for your services
3. **Monitor network traffic** - Use the Clients tab to track devices
4. **Set up WiFi hotspot** (Windows only) - Run the Python ISP engine
5. **Enable system logging** - Monitor service health and events

## Support

This is a self-contained system with no external dependencies. Everything runs locally:
- ✅ No API keys needed
- ✅ No cloud services required
- ✅ No OAuth or external authentication
- ✅ Works completely offline (except upstream DNS)
- ✅ Full control over your data

For questions, refer to the code comments and database schema in STANDALONE_README.md.

---

**You now have a fully functional personal ISP system!**
