#!/bin/bash
# Jarvis API Production Deployment Script
# This script helps automate the deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOMAIN=""
EMAIL=""
DEPLOYMENT_TYPE=""

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check for required commands
    local required_commands=("docker" "docker-compose" "curl" "openssl")
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "$cmd is not installed"
            exit 1
        fi
    done
    
    log_success "All required commands are available"
}

generate_secrets() {
    log_info "Generating security secrets..."
    
    local env_file="deployment/.env"
    
    if [[ -f "$env_file" ]]; then
        log_warning ".env file already exists. Backup and regenerate? [y/N]"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            cp "$env_file" "${env_file}.backup.$(date +%s)"
            log_info "Backed up existing .env file"
        else
            log_info "Using existing .env file"
            return
        fi
    fi
    
    # Copy template
    cp deployment/.env.example "$env_file"
    
    # Generate secrets
    local jwt_secret=$(openssl rand -base64 64)
    local admin_key=$(openssl rand -base64 32)
    local api_key_1=$(openssl rand -base64 32)
    local api_key_2=$(openssl rand -base64 32)
    local api_key_3=$(openssl rand -base64 32)
    local redis_password=$(openssl rand -base64 32)
    
    # Replace placeholders in .env file
    sed -i.bak "s|JARVIS_JWT_SECRET=.*|JARVIS_JWT_SECRET=$jwt_secret|" "$env_file"
    sed -i.bak "s|JARVIS_ADMIN_KEY=.*|JARVIS_ADMIN_KEY=$admin_key|" "$env_file"
    sed -i.bak "s|JARVIS_API_KEY_1=.*|JARVIS_API_KEY_1=$api_key_1|" "$env_file"
    sed -i.bak "s|JARVIS_API_KEY_2=.*|JARVIS_API_KEY_2=$api_key_2|" "$env_file"
    sed -i.bak "s|JARVIS_API_KEY_3=.*|JARVIS_API_KEY_3=$api_key_3|" "$env_file"
    sed -i.bak "s|REDIS_PASSWORD=.*|REDIS_PASSWORD=$redis_password|" "$env_file"
    
    if [[ -n "$DOMAIN" ]]; then
        sed -i.bak "s|DOMAIN_NAME=.*|DOMAIN_NAME=$DOMAIN|" "$env_file"
        sed -i.bak "s|your-domain.com|$DOMAIN|g" "$env_file"
    fi
    
    if [[ -n "$EMAIL" ]]; then
        sed -i.bak "s|LETSENCRYPT_EMAIL=.*|LETSENCRYPT_EMAIL=$EMAIL|" "$env_file"
        sed -i.bak "s|admin@your-domain.com|$EMAIL|g" "$env_file"
    fi
    
    # Remove backup file
    rm "${env_file}.bak"
    
    log_success "Generated security secrets in $env_file"
    
    echo
    echo "🔑 IMPORTANT: Save these API keys securely!"
    echo "Production API Key: $api_key_1"
    echo "Mobile API Key: $api_key_2"
    echo "Dashboard API Key: $api_key_3"
    echo "Admin Key: $admin_key"
    echo
}

configure_domain() {
    if [[ -z "$DOMAIN" ]]; then
        echo "Enter your domain name (e.g., api.example.com):"
        read -r DOMAIN
    fi
    
    if [[ -z "$EMAIL" ]]; then
        echo "Enter your email for SSL certificates:"
        read -r EMAIL
    fi
    
    log_info "Configuring for domain: $DOMAIN"
    
    # Update nginx configuration
    sed -i.bak "s|your-domain.com|$DOMAIN|g" deployment/nginx.conf
    sed -i.bak "s|your-email@domain.com|$EMAIL|g" deployment/docker-compose.yml
    
    # Remove backup files
    rm deployment/nginx.conf.bak deployment/docker-compose.yml.bak 2>/dev/null || true
    
    log_success "Domain configuration updated"
}

deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd deployment
    
    # Pull latest images
    docker-compose pull
    
    # Start services
    docker-compose up -d
    
    # Wait for services to start
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check health
    if curl -f http://localhost:8000/health &>/dev/null; then
        log_success "Jarvis API is running and healthy"
    else
        log_error "Jarvis API health check failed"
        docker-compose logs jarvis-api
        exit 1
    fi
    
    cd ..
    
    log_success "Docker deployment completed"
}

deploy_systemd() {
    log_info "Deploying with systemd..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        log_error "Systemd deployment requires root privileges"
        exit 1
    fi
    
    # Create user and directories
    useradd -r -s /bin/false jarvis 2>/dev/null || log_info "User jarvis already exists"
    mkdir -p /opt/jarvis /var/log/jarvis /var/lib/jarvis
    chown -R jarvis:jarvis /var/log/jarvis /var/lib/jarvis
    
    # Copy application
    cp -r . /opt/jarvis/
    chown -R jarvis:jarvis /opt/jarvis
    
    # Create virtual environment
    cd /opt/jarvis
    sudo -u jarvis python3 -m venv venv
    sudo -u jarvis venv/bin/pip install -r requirements.txt
    sudo -u jarvis venv/bin/pip install slowapi pyjwt uvicorn[standard] gunicorn
    
    # Install systemd service
    cp deployment/systemd.service /etc/systemd/system/jarvis-api.service
    systemctl daemon-reload
    systemctl enable jarvis-api
    
    # Start service
    systemctl start jarvis-api
    
    # Check status
    if systemctl is-active --quiet jarvis-api; then
        log_success "Jarvis API service is running"
    else
        log_error "Jarvis API service failed to start"
        systemctl status jarvis-api
        exit 1
    fi
    
    log_success "Systemd deployment completed"
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --docker        Deploy using Docker Compose"
    echo "  --systemd       Deploy using systemd service"
    echo "  --domain DOMAIN Set domain name"
    echo "  --email EMAIL   Set email for SSL certificates"
    echo "  --check-only    Only check requirements"
    echo "  --help          Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --docker --domain api.example.com --email admin@example.com"
    echo "  $0 --systemd"
    echo "  $0 --check-only"
}

main() {
    log_info "Jarvis API Production Deployment Script"
    echo "========================================"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --docker)
                DEPLOYMENT_TYPE="docker"
                shift
                ;;
            --systemd)
                DEPLOYMENT_TYPE="systemd"
                shift
                ;;
            --domain)
                DOMAIN="$2"
                shift 2
                ;;
            --email)
                EMAIL="$2"
                shift 2
                ;;
            --check-only)
                check_requirements
                log_success "Requirements check completed"
                exit 0
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check requirements
    check_requirements
    
    # If no deployment type specified, ask user
    if [[ -z "$DEPLOYMENT_TYPE" ]]; then
        echo "Select deployment type:"
        echo "1. Docker Compose (recommended)"
        echo "2. Systemd service"
        read -p "Enter choice [1-2]: " choice
        
        case $choice in
            1) DEPLOYMENT_TYPE="docker" ;;
            2) DEPLOYMENT_TYPE="systemd" ;;
            *) log_error "Invalid choice"; exit 1 ;;
        esac
    fi
    
    # Configure domain if not provided
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        configure_domain
    fi
    
    # Generate secrets
    generate_secrets
    
    # Deploy based on type
    case $DEPLOYMENT_TYPE in
        docker)
            deploy_docker
            ;;
        systemd)
            deploy_systemd
            ;;
        *)
            log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    echo
    log_success "🎉 Deployment completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Update DNS records to point to this server"
    echo "2. Test the API endpoints"
    echo "3. Set up monitoring and alerting"
    echo "4. Configure backup procedures"
    echo
    echo "API URL: https://$DOMAIN"
    echo "Health check: curl https://$DOMAIN/health"
    echo
    echo "Security reminders:"
    echo "- Keep API keys secure"
    echo "- Regularly update SSL certificates"
    echo "- Monitor logs for security events"
    echo "- Keep the system updated"
}

# Run main function
main "$@"