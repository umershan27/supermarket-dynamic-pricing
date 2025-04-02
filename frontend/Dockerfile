FROM node:16-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Clean install dependencies
RUN npm install

# Copy source code
COPY . .

# Set environment variables with default values
ENV REACT_APP_API_URL=http://localhost:4000/api

# Build the app
RUN npm run build

# Use nginx to serve the app
FROM nginx:alpine

# Copy the build output to replace the default nginx contents
COPY --from=build /app/build /usr/share/nginx/html

# Copy custom nginx config (to be created)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

# Start nginx
CMD ["nginx", "-g", "daemon off;"] 