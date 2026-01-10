#!/usr/bin/env bash
#
# PolyglotLink Deployment Script
#
# Usage:
#   ./scripts/deploy.sh [environment] [action]
#
# Environments: development, staging, production
# Actions: deploy, rollback, status, logs
#
# Examples:
#   ./scripts/deploy.sh staging deploy
#   ./scripts/deploy.sh production rollback
#   ./scripts/deploy.sh production status
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="polyglotlink"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/polyglotlink}"
VERSION="${VERSION:-$(git describe --tags --always 2>/dev/null || echo 'dev')}"

# Parse arguments
ENVIRONMENT="${1:-development}"
ACTION="${2:-deploy}"

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
    log_info "Checking requirements..."

    local missing=0

    if ! command -v docker &> /dev/null; then
        log_error "docker is required but not installed"
        missing=1
    fi

    if ! command -v kubectl &> /dev/null && [[ "$ENVIRONMENT" != "development" ]]; then
        log_error "kubectl is required for non-development deployments"
        missing=1
    fi

    if [[ $missing -eq 1 ]]; then
        exit 1
    fi

    log_success "All requirements met"
}

build_image() {
    log_info "Building Docker image..."

    local tag="${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}"
    local latest_tag="${DOCKER_REGISTRY}/${PROJECT_NAME}:latest"

    docker build \
        --tag "$tag" \
        --tag "$latest_tag" \
        --build-arg VERSION="$VERSION" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        .

    log_success "Built image: $tag"
}

push_image() {
    log_info "Pushing Docker image..."

    local tag="${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}"

    docker push "$tag"

    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest"
    fi

    log_success "Pushed image: $tag"
}

deploy_development() {
    log_info "Deploying to development environment..."

    # Use docker-compose for development
    docker-compose -f docker-compose.yml up -d

    log_success "Development deployment complete"
    log_info "Services available at:"
    echo "  - PolyglotLink: http://localhost:8080"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000"
}

deploy_kubernetes() {
    local namespace="${PROJECT_NAME}-${ENVIRONMENT}"

    log_info "Deploying to Kubernetes ($ENVIRONMENT)..."

    # Create namespace if it doesn't exist
    kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -

    # Apply ConfigMap
    kubectl apply -f "deploy/kubernetes/configmap-${ENVIRONMENT}.yaml" -n "$namespace"

    # Apply Secrets (assumes secrets are managed externally or exist)
    if [[ -f "deploy/kubernetes/secrets-${ENVIRONMENT}.yaml" ]]; then
        kubectl apply -f "deploy/kubernetes/secrets-${ENVIRONMENT}.yaml" -n "$namespace"
    fi

    # Apply deployment with version
    sed "s|IMAGE_TAG|${VERSION}|g" deploy/kubernetes/deployment.yaml | \
        kubectl apply -f - -n "$namespace"

    # Apply service
    kubectl apply -f deploy/kubernetes/service.yaml -n "$namespace"

    # Apply HPA for production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        kubectl apply -f deploy/kubernetes/hpa.yaml -n "$namespace"
    fi

    # Wait for rollout
    kubectl rollout status deployment/"$PROJECT_NAME" -n "$namespace" --timeout=300s

    log_success "Kubernetes deployment complete"
}

deploy() {
    check_requirements
    build_image

    case "$ENVIRONMENT" in
        development)
            deploy_development
            ;;
        staging|production)
            push_image
            deploy_kubernetes
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
}

rollback() {
    local namespace="${PROJECT_NAME}-${ENVIRONMENT}"

    log_info "Rolling back deployment in $ENVIRONMENT..."

    case "$ENVIRONMENT" in
        development)
            docker-compose down
            docker-compose up -d
            ;;
        staging|production)
            kubectl rollout undo deployment/"$PROJECT_NAME" -n "$namespace"
            kubectl rollout status deployment/"$PROJECT_NAME" -n "$namespace" --timeout=300s
            ;;
    esac

    log_success "Rollback complete"
}

status() {
    log_info "Checking deployment status..."

    case "$ENVIRONMENT" in
        development)
            docker-compose ps
            ;;
        staging|production)
            local namespace="${PROJECT_NAME}-${ENVIRONMENT}"
            echo ""
            echo "=== Pods ==="
            kubectl get pods -n "$namespace" -o wide
            echo ""
            echo "=== Services ==="
            kubectl get svc -n "$namespace"
            echo ""
            echo "=== Deployments ==="
            kubectl get deployments -n "$namespace"
            ;;
    esac
}

logs() {
    log_info "Fetching logs..."

    case "$ENVIRONMENT" in
        development)
            docker-compose logs -f --tail=100 polyglotlink
            ;;
        staging|production)
            local namespace="${PROJECT_NAME}-${ENVIRONMENT}"
            kubectl logs -f -l app="$PROJECT_NAME" -n "$namespace" --tail=100
            ;;
    esac
}

# Main
echo ""
echo "========================================"
echo "  PolyglotLink Deployment"
echo "  Environment: $ENVIRONMENT"
echo "  Action: $ACTION"
echo "  Version: $VERSION"
echo "========================================"
echo ""

case "$ACTION" in
    deploy)
        deploy
        ;;
    rollback)
        rollback
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        log_error "Unknown action: $ACTION"
        echo "Usage: $0 [environment] [deploy|rollback|status|logs]"
        exit 1
        ;;
esac
