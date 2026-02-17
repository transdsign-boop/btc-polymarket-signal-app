FROM node:20-alpine AS frontend-build

WORKDIR /frontend

COPY frontend/package.json /frontend/package.json
RUN npm install
COPY frontend /frontend
ARG VITE_API_BASE=""
ENV VITE_API_BASE=${VITE_API_BASE}
RUN npm run build

FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend /app
COPY --from=frontend-build /frontend/dist /app/frontend_dist

ENV PORT=8080
ENV FRONTEND_DIST_DIR=/app/frontend_dist
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
