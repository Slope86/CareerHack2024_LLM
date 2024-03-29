# Basic Image Environment
FROM python:3.11.7

# 指定 Image 中的工作目錄
WORKDIR /code

# 將 Dockerfile 所在目錄下的所有檔案複製到 Image 的工作目錄 /code 底下
ADD . /code

# 在 Image 中執行的指令：安裝 requirements.txt 中所指定的 dependencies
RUN pip install -r requirements.txt

# Container 啟動指令：Container 啟動後通過 python 運行 api server
CMD ["python", "LLMapi.py"]