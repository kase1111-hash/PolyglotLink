# PolyglotLink Makefile
# Automated build scripts for development, testing, and deployment

.PHONY: help install install-dev test test-unit test-integration test-cov lint format type-check security-check static-analysis clean build docker-build docker-up docker-down run dev docs

# Default target
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python3
PIP := pip3
PYTEST := pytest
PROJECT_NAME := polyglotlink

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

#-------------------------------------------------------------------------------
# Help
#-------------------------------------------------------------------------------

help: ## Show this help message
	@echo "$(BLUE)PolyglotLink - Semantic API Translator for IoT$(NC)"
	@echo ""
	@echo "$(YELLOW)Usage:$(NC)"
	@echo "  make [target]"
	@echo ""
	@echo "$(YELLOW)Targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

#-------------------------------------------------------------------------------
# Installation
#-------------------------------------------------------------------------------

install: ## Install production dependencies
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev,test]"
	pre-commit install || true

install-all: ## Install all dependencies including optional ones
	$(PIP) install -e ".[dev,test,docs]"
	pre-commit install || true

#-------------------------------------------------------------------------------
# Testing
#-------------------------------------------------------------------------------

test: ## Run all tests
	$(PYTEST) $(PROJECT_NAME)/tests/ -v

test-unit: ## Run unit tests only
	$(PYTEST) $(PROJECT_NAME)/tests/ -v -m "not integration" --ignore=$(PROJECT_NAME)/tests/test_integration.py

test-integration: ## Run integration tests only
	$(PYTEST) $(PROJECT_NAME)/tests/test_integration.py -v

test-cov: ## Run tests with coverage report
	$(PYTEST) $(PROJECT_NAME)/tests/ -v --cov=$(PROJECT_NAME) --cov-report=term-missing --cov-report=html --cov-report=xml

test-fast: ## Run tests in parallel (requires pytest-xdist)
	$(PYTEST) $(PROJECT_NAME)/tests/ -v -n auto

test-watch: ## Run tests in watch mode (requires pytest-watch)
	ptw $(PROJECT_NAME)/tests/ -- -v

#-------------------------------------------------------------------------------
# Code Quality
#-------------------------------------------------------------------------------

lint: ## Run linting checks
	@echo "$(BLUE)Running ruff linter...$(NC)"
	ruff check $(PROJECT_NAME)/
	@echo "$(GREEN)Linting passed!$(NC)"

lint-fix: ## Run linter and fix issues automatically
	@echo "$(BLUE)Running ruff linter with auto-fix...$(NC)"
	ruff check $(PROJECT_NAME)/ --fix
	@echo "$(GREEN)Linting fixed!$(NC)"

format: ## Format code with ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	ruff format $(PROJECT_NAME)/
	@echo "$(GREEN)Formatting complete!$(NC)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code format...$(NC)"
	ruff format $(PROJECT_NAME)/ --check
	@echo "$(GREEN)Format check passed!$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy $(PROJECT_NAME)/ --ignore-missing-imports
	@echo "$(GREEN)Type checking passed!$(NC)"

security-check: ## Run security vulnerability scan
	@echo "$(BLUE)Running security scan with bandit...$(NC)"
	bandit -r $(PROJECT_NAME)/ -ll -ii
	@echo "$(BLUE)Checking dependencies with safety...$(NC)"
	safety check || pip-audit
	@echo "$(GREEN)Security checks passed!$(NC)"

static-analysis: lint format-check type-check security-check ## Run all static analysis checks
	@echo "$(GREEN)All static analysis checks passed!$(NC)"

#-------------------------------------------------------------------------------
# Build & Package
#-------------------------------------------------------------------------------

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf $(PROJECT_NAME)/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Clean complete!$(NC)"

build: clean ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)Build complete!$(NC)"

build-wheel: ## Build wheel package only
	@echo "$(BLUE)Building wheel package...$(NC)"
	$(PYTHON) -m build --wheel
	@echo "$(GREEN)Wheel build complete!$(NC)"

#-------------------------------------------------------------------------------
# Docker
#-------------------------------------------------------------------------------

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(PROJECT_NAME):latest .
	@echo "$(GREEN)Docker build complete!$(NC)"

docker-build-dev: ## Build Docker image for development
	@echo "$(BLUE)Building development Docker image...$(NC)"
	docker build -t $(PROJECT_NAME):dev --target builder .
	@echo "$(GREEN)Development Docker build complete!$(NC)"

docker-up: ## Start all services with docker-compose
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Services started!$(NC)"

docker-down: ## Stop all services
	@echo "$(BLUE)Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)Services stopped!$(NC)"

docker-logs: ## View service logs
	docker-compose logs -f

docker-clean: ## Remove all containers and volumes
	@echo "$(YELLOW)Warning: This will remove all containers and volumes!$(NC)"
	docker-compose down -v --remove-orphans
	docker system prune -f

#-------------------------------------------------------------------------------
# Development
#-------------------------------------------------------------------------------

run: ## Run the application
	$(PYTHON) -m $(PROJECT_NAME).app.main serve

dev: ## Run in development mode with auto-reload
	$(PYTHON) -m $(PROJECT_NAME).app.main serve --verbose

check: ## Run health check
	$(PYTHON) -m $(PROJECT_NAME).app.main check

version: ## Show version
	$(PYTHON) -m $(PROJECT_NAME).app.main version

#-------------------------------------------------------------------------------
# Documentation
#-------------------------------------------------------------------------------

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	cd docs && make html || echo "$(YELLOW)Documentation not configured$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	cd docs/_build/html && python -m http.server 8000 || echo "$(YELLOW)Build docs first with 'make docs'$(NC)"

#-------------------------------------------------------------------------------
# Database
#-------------------------------------------------------------------------------

db-migrate: ## Run database migrations (if applicable)
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head || echo "$(YELLOW)No migrations configured$(NC)"

db-rollback: ## Rollback last migration
	@echo "$(BLUE)Rolling back last migration...$(NC)"
	alembic downgrade -1 || echo "$(YELLOW)No migrations configured$(NC)"

#-------------------------------------------------------------------------------
# CI/CD Helpers
#-------------------------------------------------------------------------------

ci-test: ## Run CI test suite
	$(PYTEST) $(PROJECT_NAME)/tests/ -v --cov=$(PROJECT_NAME) --cov-report=xml --cov-fail-under=70

ci-lint: ## Run CI linting
	ruff check $(PROJECT_NAME)/ --output-format=github

ci-all: ci-lint ci-test ## Run all CI checks
	@echo "$(GREEN)All CI checks passed!$(NC)"

#-------------------------------------------------------------------------------
# Release
#-------------------------------------------------------------------------------

release-patch: ## Create a patch release (x.x.X)
	@echo "$(BLUE)Creating patch release...$(NC)"
	bump2version patch
	@echo "$(GREEN)Patch release created!$(NC)"

release-minor: ## Create a minor release (x.X.0)
	@echo "$(BLUE)Creating minor release...$(NC)"
	bump2version minor
	@echo "$(GREEN)Minor release created!$(NC)"

release-major: ## Create a major release (X.0.0)
	@echo "$(BLUE)Creating major release...$(NC)"
	bump2version major
	@echo "$(GREEN)Major release created!$(NC)"

#-------------------------------------------------------------------------------
# Deployment
#-------------------------------------------------------------------------------

deploy-dev: ## Deploy to development environment
	@echo "$(BLUE)Deploying to development...$(NC)"
	./scripts/deploy.sh development deploy

deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(NC)"
	./scripts/deploy.sh staging deploy

deploy-prod: ## Deploy to production environment
	@echo "$(YELLOW)Deploying to production...$(NC)"
	./scripts/deploy.sh production deploy

rollback-staging: ## Rollback staging deployment
	@echo "$(YELLOW)Rolling back staging...$(NC)"
	./scripts/deploy.sh staging rollback

rollback-prod: ## Rollback production deployment
	@echo "$(RED)Rolling back production...$(NC)"
	./scripts/deploy.sh production rollback

deploy-status: ## Check deployment status
	./scripts/deploy.sh $(ENV) status

deploy-logs: ## View deployment logs
	./scripts/deploy.sh $(ENV) logs

#-------------------------------------------------------------------------------
# Load Testing
#-------------------------------------------------------------------------------

test-load: ## Run load tests with Locust (web UI)
	@echo "$(BLUE)Starting Locust load test...$(NC)"
	locust -f $(PROJECT_NAME)/tests/performance/locustfile.py

test-load-headless: ## Run headless load test
	@echo "$(BLUE)Running headless load test...$(NC)"
	locust -f $(PROJECT_NAME)/tests/performance/locustfile.py \
		--headless -u 100 -r 10 -t 60s \
		--host http://localhost:8080 \
		--html=load_test_report.html
