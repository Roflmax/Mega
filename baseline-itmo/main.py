import time
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger
import httpx
import json
from openai import OpenAI
import xml.etree.ElementTree as ET

# Initialize
app = FastAPI()
logger = None
client = OpenAI(
    api_key="sk-4c4a47f9b4114cb7b223a73bd4778bc1",  # Ваш API-ключ DeepSeek
    base_url="https://api.deepseek.com",  # URL API DeepSeek
)

@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()

# Разрешить все источники (для тестирования)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_llm_answer(query: str, context: str) -> dict:
    """
    Запрос к языковой модели DeepSeek для генерации ответа.
    """
    system_prompt = """
    Ты эксперт по Университету ИТМО. Отвечай на вопросы чётко и информативно.
    Если есть варианты ответов, выбери номер (1-10) и объясни свой выбор в reasoning.
    Если вариантов ответа в вопросе нет,то в answer запиши "null", но в reasoning дай развёрнутый ответ!
    #### Пример ввода:
    {
        "query": "В каком городе находится главный кампус Университета ИТМО?\n1. Москва\n2. Санкт-Петербург\n3. Екатеринбург\n4. Нижний Новгород",
    }
    #### Пример вывода в формате JSON:
    {
        "answer": "2",
        "reasoning": "Университет Итмо находится в Санкт-Петербурге ....(пояснение)"
    }

        #### Пример ввода:
    {
        "query": "В каком городе находится главный кампус Университета ИТМО?",
    }
    #### Пример вывода в формате JSON:
    {
        "answer": "null",
        "reasoning": "Университет Итмо находится в Санкт-Петербурге ....(пояснение)"
    }
    """
    user_prompt = f"""
    Вопрос: {query}
    Контекст: {context}
    Ответ в формате JSON: {{ "answer": номер или null, "reasoning": "текст" }}
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        try:
            parsed = json.loads(content)
            answer = parsed.get('answer')
            if answer == 'null':
                answer = None
            reasoning = parsed.get('reasoning', '') + " Ответ сгенерирован моделью deepseek."
            return {'answer': answer, 'reasoning': reasoning}
        except Exception as e:
            return {'answer': None, 'reasoning': f"Ошибка обработки ответа модели: {str(e)}"}
    except Exception as e:
        return {'answer': None, 'reasoning': f"Ошибка модели: {str(e)}"}

async def search_sources(query: str) -> list:
    """
    Поиск информации через Yandex Search API.
    """
    api_key = "AQVN2iM2CEvwpWixltOG3bhRvRhwzTGKKHfWUo4K"
    user_id = "b1g67dbhtsk2c1ke5k9b"
    url = "https://yandex.ru/search/xml"
    request_body = f"""
    <request>
        <query>{query}</query>
        <sortby>rlv</sortby>
        <maxpassages>3</maxpassages>
        <page>0</page>
        <groupings>
            <groupby attr="d" mode="deep" groups-on-page="3" docs-in-group="1"/>
        </groupings>
    </request>
    """.strip()
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/xml"
    }
    params = {
        "folderid": user_id,
        "lr": 2  
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                params=params,
                headers=headers,
                content=request_body
            )
            
            if response.status_code != 200:
                return [], []

            root = ET.fromstring(response.text)
            links = []
            text = []
            for passage in root.findall(".//extended-text"):
                if passage.text:
                    text.append(passage.text)
            for url_tag in root.findall(".//url"):
                if url_tag.text:
                    links.append(url_tag.text.strip())
            
            return text[:3], links[:3]
        
        except ET.ParseError as e:
            return [], []
        except Exception as e:
            return [], []

async def get_news() -> list:
    """
    Парсинг новостей с RSS-ленты ИТМО.
    """
    feed = feedparser.parse("https://news.itmo.ru/ru/rss/")
    return [entry.link for entry in feed.entries[:3]]

# Middleware для логирования
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    body = await request.body()
    
    try:
        await logger.info(f"Incoming request: {request.method} {request.url}\nRequest body: {body.decode()}")
    except Exception as e:
        await logger.error(f"Failed to log request: {str(e)}")

    response = await call_next(request)
    process_time = time.time() - start_time
    
    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk
    
    logger.info(f"Request completed: {request.method} {request.url}\nStatus: {response.status_code}\nResponse body: {response_body.decode()}\nDuration: {process_time:.3f}s")
    
    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

# Обработка корневого URL
@app.get("/")
async def root():
    return {"message": "Welcome to the API"}

# Обработка favicon
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

# Основной эндпоинт
@app.post("/api/request", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        await logger.info(f"Processing prediction request with id: {request.id}")
        
        # Поиск источников и извлечение текста
        text, sources = await search_sources(request.query)
        context = "\n".join(text)

        # Генерация ответа моделью с использованием найденного контекста
        res_llm = await get_llm_answer(request.query, context)
        answer = res_llm.get('answer')
        reasoning = res_llm.get('reasoning')+" |Deepseek"
        
        # Формирование ответа
        response = PredictionResponse(
            id=request.id,
            answer=answer,  
            reasoning=reasoning,
            sources=sources
        )
        
        await logger.info(f"Successfully processed request {request.id}")
        return response
    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {request.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {request.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)