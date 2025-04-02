.PHONY: build test up down

build:
	docker-compose build

test:
	cd frontend && npm test
	cd backend && npm test
	cd ml-service && pytest

up:
	docker-compose up -d

down:
	docker-compose down 