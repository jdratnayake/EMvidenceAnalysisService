# Use the official Python image as base
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y libhdf5-dev
RUN pip install --upgrade pip setuptools
RUN pip install h5py
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 80 to the outside world
EXPOSE 80

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
