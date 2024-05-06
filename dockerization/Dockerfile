FROM  nvcr.io/nvidia/pytorch:24.02-py3
WORKDIR /workspace/project
RUN addgroup --gid 4321 vscode
RUN adduser --uid 1234 --gid 4321 vscode
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt