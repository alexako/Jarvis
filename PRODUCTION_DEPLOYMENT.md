# Jarvis API Production Deployment Guide

This guide provides comprehensive instructions for securely deploying the Jarvis API to production environments.

## 🔒 Security Architecture

### Multi-Layer Security Approach

1. **Network Security**
   - HTTPS/TLS encryption (TLS 1.2+)
   - Reverse proxy with Nginx
   - Rate limiting and DDoS protection
   - IP allowlisting/blocklisting
   - Firewall configuration

2. **Application Security**
   - JWT token authentication
   - API key management
   - Permission-based authorization
   - Request validation and sanitization
   - Security headers

3. **Infrastructure Security**
   - Container isolation
   - Non-root user execution
   - Resource limits
   - Read-only filesystems where possible
   - Security monitoring

## 🚀 Deployment Options

### Option 1: Docker Compose (Recommended)

**Prerequisites:**
- Docker Engine 20.10+
- Docker Compose 2.0+
- Domain name with DNS configured

**Setup:**

1. **Clone and prepare:**
```bash
git clone <your-repo>
cd jarvis/deployment
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Generate secrets:**
```bash
# Generate JWT secret
openssl rand -base64 64

# Generate API keys
openssl rand -base64 32

# Generate admin key
openssl rand -base64 32
```

4. **Update domain settings:**
```bash
# Edit docker-compose.yml and nginx.conf
# Replace 'your-domain.com' with your actual domain
```

5. **Deploy:**
```bash
docker-compose up -d
```

### Option 2: Systemd Service

**Prerequisites:**
- Ubuntu 20.04+ / CentOS 8+ / Similar Linux distribution
- Python 3.12+
- Nginx
- Certbot (for SSL certificates)

**Setup:**

1. **Create service user:**
```bash
sudo useradd -r -s /bin/false jarvis
sudo mkdir -p /opt/jarvis /var/log/jarvis /var/lib/jarvis
sudo chown jarvis:jarvis /var/log/jarvis /var/lib/jarvis
```

2. **Install application:**
```bash
sudo cp -r . /opt/jarvis/
sudo chown -R jarvis:jarvis /opt/jarvis
cd /opt/jarvis
sudo -u jarvis python -m venv venv
sudo -u jarvis venv/bin/pip install -r requirements.txt
```

3. **Install system service:**
```bash
sudo cp deployment/systemd.service /etc/systemd/system/jarvis-api.service
sudo systemctl daemon-reload
sudo systemctl enable jarvis-api
```

4. **Configure Nginx:**
```bash
sudo cp deployment/nginx.conf /etc/nginx/sites-available/jarvis-api
sudo ln -s /etc/nginx/sites-available/jarvis-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

5. **Get SSL certificate:**
```bash
sudo certbot --nginx -d your-domain.com
```

6. **Start service:**
```bash
sudo systemctl start jarvis-api
```

## 🔐 Security Configuration

### Environment Variables

**Required Security Variables:**
```bash
# JWT Secret (64+ random characters)
JARVIS_JWT_SECRET=your-jwt-secret-here

# API Keys (32+ random characters each)
JARVIS_API_KEY_1=your-api-key-here
JARVIS_API_KEY_1_NAME=production_client
JARVIS_API_KEY_1_PERMISSIONS=read,chat,stream

# Admin access
JARVIS_ADMIN_KEY=your-admin-key-here

# CORS and trusted hosts
JARVIS_ALLOWED_ORIGINS=https://your-domain.com,https://app.your-domain.com
JARVIS_TRUSTED_HOSTS=your-domain.com,admin.your-domain.com

# Optional: IP restrictions
JARVIS_ALLOWED_IPS=192.168.1.0/24,10.0.0.0/8
JARVIS_BLOCKED_IPS=1.2.3.4,5.6.7.8
```

**AI Provider Keys:**
```bash
ANTHROPIC_API_KEY=your-anthropic-key
DEEPSEEK_API_KEY=your-deepseek-key
```

### Permission System

The API supports role-based permissions:

- **read**: Access to status and health endpoints
- **chat**: Access to chat and conversation endpoints  
- **stream**: Access to audio streaming endpoints
- **admin**: Access to administrative functions

### API Key Management

**Creating API Keys:**
```bash
# Generate new API key
openssl rand -base64 32

# Add to environment
JARVIS_API_KEY_2=new-key-here
JARVIS_API_KEY_2_NAME=mobile_app
JARVIS_API_KEY_2_PERMISSIONS=read,chat
```

**Using API Keys:**
```bash
# In Authorization header
curl -H "Authorization: Bearer your-api-key-here" \
     https://your-domain.com/status
```

## 🛡️ Firewall Configuration

### UFW (Ubuntu)
```bash
# Reset firewall
sudo ufw --force reset

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH access
sudo ufw allow ssh

# HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Optional: Restrict SSH to specific IPs
sudo ufw delete allow ssh
sudo ufw allow from 192.168.1.0/24 to any port 22

# Enable firewall
sudo ufw enable
```

### iptables (Manual)
```bash
# Basic firewall rules
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (restrict to your IP)
iptables -A INPUT -p tcp --dport 22 -s YOUR_IP_HERE -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Rate limiting
iptables -A INPUT -p tcp --dport 80 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT

# Save rules
iptables-save > /etc/iptables/rules.v4
```

## 📊 Monitoring and Logging

### Log Locations

**Docker Deployment:**
```bash
# Container logs
docker logs jarvis-api
docker logs jarvis-nginx

# Application logs
docker exec jarvis-api tail -f /app/logs/api.log
```

**Systemd Deployment:**
```bash
# Service logs
journalctl -u jarvis-api -f

# Application logs
tail -f /var/log/jarvis/api.log

# Nginx logs
tail -f /var/log/nginx/jarvis-api.access.log
tail -f /var/log/nginx/jarvis-api.error.log
```

### Security Event Monitoring

Monitor for these security events:
- Failed authentication attempts
- Rate limit violations
- IP blocking events
- Unusual request patterns
- Large file uploads

**Log Analysis:**
```bash
# Failed auth attempts
grep "SECURITY_EVENT" /var/log/jarvis/api.log | grep "auth_failed"

# Rate limit violations
grep "429" /var/log/nginx/jarvis-api.access.log

# IP patterns
awk '{print $1}' /var/log/nginx/jarvis-api.access.log | sort | uniq -c | sort -nr
```

### Health Monitoring

**Automated Health Checks:**
```bash
# Create monitoring script
cat > /usr/local/bin/jarvis-health-check << 'EOF'
#!/bin/bash
response=$(curl -s -o /dev/null -w "%{http_code}" https://your-domain.com/health)
if [ "$response" != "200" ]; then
    echo "ALERT: Jarvis API health check failed (HTTP $response)"
    # Send alert (email, Slack, etc.)
fi
EOF

chmod +x /usr/local/bin/jarvis-health-check

# Add to crontab
echo "*/5 * * * * /usr/local/bin/jarvis-health-check" | crontab -
```

## 🔄 Maintenance and Updates

### SSL Certificate Renewal

**Automatic (with certbot):**
```bash
# Test renewal
sudo certbot renew --dry-run

# Renewal happens automatically via cron
```

**Docker deployment:**
- Certificates auto-renew via the certbot container

### Application Updates

**Docker Deployment:**
```bash
# Pull latest images
docker-compose pull

# Restart with zero downtime
docker-compose up -d --no-deps jarvis-api
```

**Systemd Deployment:**
```bash
# Update code
sudo systemctl stop jarvis-api
sudo cp -r new-version/* /opt/jarvis/
sudo chown -R jarvis:jarvis /opt/jarvis
sudo systemctl start jarvis-api
```

### Backup Strategy

**Critical Data:**
- Database: `/var/lib/jarvis/jarvis_memory.db`
- Configuration: Environment variables
- SSL certificates: `/etc/letsencrypt/`
- Logs: `/var/log/jarvis/`

**Backup Script:**
```bash
#!/bin/bash
BACKUP_DIR="/backup/jarvis/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Database
cp /var/lib/jarvis/jarvis_memory.db "$BACKUP_DIR/"

# Configuration
env | grep JARVIS_ > "$BACKUP_DIR/environment.txt"

# SSL certificates
tar -czf "$BACKUP_DIR/ssl-certs.tar.gz" /etc/letsencrypt/

# Logs (last 7 days)
find /var/log/jarvis/ -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/" \;

echo "Backup completed: $BACKUP_DIR"
```

## 🚨 Security Incident Response

### Immediate Actions

1. **Identify the threat:**
   - Check logs for unusual patterns
   - Monitor resource usage
   - Verify API key usage

2. **Contain the incident:**
   - Block malicious IPs
   - Revoke compromised API keys
   - Scale down if under attack

3. **Investigate:**
   - Review audit logs
   - Check for data access
   - Analyze attack vectors

4. **Recover:**
   - Update security configurations
   - Regenerate compromised secrets
   - Apply security patches

### Incident Commands

**Block IP immediately:**
```bash
# UFW
sudo ufw insert 1 deny from IP_ADDRESS

# iptables
iptables -I INPUT -s IP_ADDRESS -j DROP
```

**Revoke API key:**
```bash
# Remove from environment
unset JARVIS_API_KEY_X

# Restart service
sudo systemctl restart jarvis-api
```

**Emergency shutdown:**
```bash
# Stop all services
sudo systemctl stop jarvis-api nginx

# Or with Docker
docker-compose down
```

## 📞 Support and Troubleshooting

### Common Issues

**Connection Refused:**
- Check firewall rules
- Verify service status
- Check port bindings

**SSL Certificate Errors:**
- Verify certificate validity
- Check domain DNS settings
- Review nginx configuration

**High CPU/Memory Usage:**
- Check for request patterns
- Review rate limiting
- Monitor concurrent connections

**Authentication Failures:**
- Verify API key format
- Check permission settings
- Review JWT token expiration

### Getting Help

For production support:
1. Check logs first
2. Review configuration
3. Test with minimal setup
4. Contact system administrator

---

**Important:** Always test deployments in a staging environment before production. This guide provides a foundation, but security requirements vary by organization and use case.