# Use an official Node runtime as the builder stage
FROM node:20.11.1-slim as builder

# Set working directory for this stage
WORKDIR /usr/src/app

# Copy package.json and package-lock.json for installing dependencies
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy your static assets and Tailwind config files
COPY ./myapp/static ./myapp/static
COPY tailwind.config.js ./

# Run the Tailwind build process
# RUN npx tailwindcss-cli build myapp/static/src/input.css -o myapp/static/src/output.css

# Use an official Python runtime as the final stage
FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /usr/src/app

COPY ./wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Install Python dependencies
COPY ./requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Update the package list
RUN apt-get update

# Install libsndfile
RUN apt-get install -y libsndfile1


# Copy project files from the builder stage and the current directory
COPY --from=builder /usr/src/app/myapp/static ./myapp/static
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

