## Агент для Iris

В рамках пет-проекта я разработал простого консольного ИИ-агента для классификации ирисов по четырём параметрам, используя датасет Iris с Kaggle. Основной упор при этом был сделан на изучение архитектуры и принципов работы ИИ‑агента 

## Структура
- `Iris.ipynb` — ноутбук с обучением модели
- `agent.py` — логика агента
- `run.py` — скрипт
- `iris_model.pkl` — сохранённая модель

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Настройки (переменные окружения)
Для использования LLM нужен `OPENAI_API_KEY`.

Опционально можно переопределить:
- `OPENAI_API_BASE` (по умолчанию: `https://api.vsegpt.ru/v1`)
- `OPENAI_MODEL` (по умолчанию: `gpt-4o-mini`)
Пример для PowerShell (только на текущую сессию):

```powershell
$env:OPENAI_API_KEY = "ваш_ключ"
$env:OPENAI_API_BASE = "https://api.vsegpt.ru/v1"
$env:OPENAI_MODEL = "gpt-4o-mini"
```

## Запуск

```powershell
python run.py
```

Примеры ввода:
- Только числа (4 параметра): `5.1 3.5 1.4 0.2`
- Фраза + числа: `определи вид ириса 3.2 1.3 3.4 4.5`

