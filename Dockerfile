# Use the official Python 3.12 image from the Docker Hub
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY . .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "/app/main.py"]
