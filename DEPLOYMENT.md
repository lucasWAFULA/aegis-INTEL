# Deployment Guide

This guide covers deploying the ML-TSSP HUMINT Dashboard to production environments.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment Options](#deployment-options)
- [Cloud Deployment](#cloud-deployment)
- [Docker Deployment](#docker-deployment)
- [Security Hardening](#security-hardening)
- [Monitoring & Maintenance](#monitoring--maintenance)

---

## Prerequisites

### System Requirements

**Minimum:**
- 2 vCPUs
- 4GB RAM
- 20GB disk space
- Ubuntu 20.04+ / RHEL 8+ / Windows Server 2019+

**Recommended:**
- 4 vCPUs
- 8GB RAM
- 50GB disk space
- Load balancer for high availability

### Software Requirements

- Python 3.8+
- pip 21.0+
- Git
- SSL/TLS certificates
- Reverse proxy (Nginx/Apache)

---

## Deployment Options

### Option 1: Traditional Server Deployment

#### 1. Prepare Server

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.9 python3.9-venv python3-pip nginx certbot

# Create application user
sudo useradd -m -s /bin/bash humint
sudo su - humint
```

#### 2. Deploy Application

```bash
# Clone repository
git clone https://github.com/yourusername/humint-dashboard.git
cd humint-dashboard

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Edit configuration
```

#### 3. Configure Systemd Service

Create `/etc/systemd/system/humint-dashboard.service`:

```ini
[Unit]
Description=ML-TSSP HUMINT Dashboard
After=network.target

[Service]
Type=simple
User=humint
WorkingDirectory=/home/humint/humint-dashboard
Environment="PATH=/home/humint/humint-dashboard/venv/bin"
ExecStart=/home/humint/humint-dashboard/venv/bin/streamlit run dashboard.py --server.port=8501 --server.address=127.0.0.1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable humint-dashboard
sudo systemctl start humint-dashboard
sudo systemctl status humint-dashboard
```

#### 4. Configure Nginx Reverse Proxy

Create `/etc/nginx/sites-available/humint-dashboard`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Proxy settings
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Increase buffer sizes for large requests
    client_max_body_size 50M;
    client_body_buffer_size 1M;
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/humint-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 5. Obtain SSL Certificate

```bash
sudo certbot --nginx -d your-domain.com
sudo systemctl reload nginx
```

---

### Option 2: Docker Deployment

#### 1. Create Dockerfile

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  dashboard:
    build: .
    image: humint-dashboard:latest
    container_name: humint-dashboard
    restart: unless-stopped
    ports:
      - "8501:8501"
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - DEBUG_MODE=False
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    networks:
      - humint-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  humint-network:
    driver: bridge
```

#### 3. Build and Run

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

### Option 3: Cloud Deployment

#### AWS Deployment (EC2)

1. **Launch EC2 Instance**
   - Instance type: t3.medium or larger
   - AMI: Ubuntu 20.04 LTS
   - Security group: Allow ports 80, 443, 22

2. **Configure Security Group**
```bash
# Inbound rules
22    SSH      Your IP
80    HTTP     0.0.0.0/0
443   HTTPS    0.0.0.0/0
```

3. **Deploy Application**
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Follow traditional deployment steps
```

#### Google Cloud Platform (Cloud Run)

1. **Prepare for Cloud Run**

Create `app.yaml`:
```yaml
runtime: python39
instance_class: F2

env_variables:
  SECRET_KEY: 'your-secret-key'
  DEBUG_MODE: 'False'

handlers:
- url: /.*
  script: auto
```

2. **Deploy**
```bash
gcloud run deploy humint-dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure (App Service)

```bash
# Create resource group
az group create --name humint-rg --location eastus

# Create App Service plan
az appservice plan create --name humint-plan \
  --resource-group humint-rg --sku B2 --is-linux

# Create web app
az webapp create --resource-group humint-rg \
  --plan humint-plan --name humint-dashboard \
  --runtime "PYTHON|3.9"

# Deploy code
az webapp up --name humint-dashboard \
  --resource-group humint-rg
```

---

## Security Hardening

### 1. Environment Variables

Never commit secrets to Git. Use environment variables:

```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set in .env
SECRET_KEY=your-generated-key
```

### 2. Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 3. Database Encryption

If using a database:
```python
# Use encrypted connections
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
```

### 4. Rate Limiting

Configure in Nginx:
```nginx
limit_req_zone $binary_remote_addr zone=dashboard_limit:10m rate=10r/s;

server {
    location / {
        limit_req zone=dashboard_limit burst=20 nodelay;
        # ... other config
    }
}
```

### 5. Authentication

Replace demo credentials with production auth:
```python
# Use OAuth 2.0, SAML, or LDAP
# Example with Azure AD
from streamlit_oauth import OAuth2Component

oauth2 = OAuth2Component(
    client_id="your-client-id",
    client_secret="your-client-secret",
    authorize_endpoint="https://login.microsoftonline.com/.../oauth2/v2.0/authorize"
)
```

---

## Monitoring & Maintenance

### 1. Application Logging

Configure logging in `dashboard.py`:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/humint/dashboard.log'),
        logging.StreamHandler()
    ]
)
```

### 2. Health Checks

Endpoint for monitoring:
```python
@st.cache_data
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
```

### 3. Automated Backups

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/humint"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup data
tar -czf $BACKUP_DIR/data_$DATE.tar.gz /home/humint/humint-dashboard/data

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz /home/humint/humint-dashboard/models

# Keep only last 7 days
find $BACKUP_DIR -type f -mtime +7 -delete
```

Schedule with cron:
```cron
0 2 * * * /home/humint/backup.sh
```

### 4. Performance Monitoring

Use monitoring tools:
- **Prometheus + Grafana**: Metrics and dashboards
- **ELK Stack**: Log aggregation and analysis
- **New Relic / Datadog**: APM and alerting

### 5. Updates and Patches

```bash
# Automated updates script
#!/bin/bash
cd /home/humint/humint-dashboard

# Pull latest code
git pull origin main

# Activate venv
source venv/bin/activate

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl restart humint-dashboard
```

---

## Troubleshooting

### Common Issues

**Issue: Port already in use**
```bash
# Find process using port
sudo lsof -i :8501
# Kill process
sudo kill -9 <PID>
```

**Issue: Permission denied**
```bash
# Fix ownership
sudo chown -R humint:humint /home/humint/humint-dashboard
# Fix permissions
chmod -R 755 /home/humint/humint-dashboard
```

**Issue: SSL certificate errors**
```bash
# Renew certificate
sudo certbot renew
sudo systemctl reload nginx
```

---

## Support

For deployment issues:
- Check logs: `journalctl -u humint-dashboard -f`
- Review Nginx logs: `tail -f /var/log/nginx/error.log`
- Open GitHub issue with deployment details

---

**🛡️ ML-TSSP HUMINT Dashboard - Deployment Guide v1.0.0**
