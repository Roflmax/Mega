FROM python:3.9-slim
WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода (будет монтироваться через volume)
COPY . .

# Запуск bash скрипта
RUN chmod +x start.sh

CMD ["sh", "./start.sh"]