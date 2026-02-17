FROM node:20-alpine AS build
WORKDIR /app

COPY package.json /app/package.json
RUN npm install

COPY . /app
ARG VITE_API_BASE=https://REPLACE_WITH_BACKEND_HOST
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
