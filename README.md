# Dynamic Pricing System

A microservices-based system for predicting retail prices using machine learning.

## Prerequisites

- Docker
- Docker Compose
- Make

## Getting Started

1. Build all services:
```bash
make build
```

2. Run tests:
```bash
make test
```

3. Start the system:
```bash
make up
```

4. Stop the system:
```bash
make down
```

## Services

- Frontend: http://localhost:3000
- Backend: http://localhost:4000
- ML Service: http://localhost:5000

## Architecture

This project follows hexagonal (ports & adapters) architecture and Domain-Driven Design principles. 