FROM node:20

WORKDIR /app

COPY package*.json ./

# Install all dependencies including devDependencies for TypeScript compilation
RUN npm install

COPY . .

# Make sure the directory exists
RUN mkdir -p dist

# Compile TypeScript
RUN npm run build

EXPOSE 4000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:4000/health || exit 1

CMD ["node", "dist/app.js"] 