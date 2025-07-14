# 🏦 Bank Reviews Classification + FastAPI

Проект: от обучения NLP-модели до интеграции её в REST API на FastAPI для банковских отзывов.

## 📋 Описание
Клиенты описывают ситуацию, а модель автоматически определяет её категорию:
- `финансовая`
- `мошенничество`
- `логистика`
- `обслуживание`
- `прочее`

## 🔧 Технологии
- Python 3
- Pandas, Scikit-learn, NLTK
- Multinomial Naive Bayes
- FastAPI, Uvicorn

## 🚀 Возможности
✅ Определение категории проблемы клиента  
✅ REST API на FastAPI для приёма и классификации отзывов

## 📦 Состав
- `reviews_NLP.ipynb` — обучение и сохранение модели
- `model_reviews.pkl` — обученная модель
- `vec_reviews.pkl` — векторизатор текста
- `app.py` — FastAPI приложение

Установи зависимости: pip install -r requirements.txt
Запусти сервер FastAPI:uvicorn app:app --reload
Отправь POST-запрос на /reviews/:
{
  "text": "Оплатил — не получил сегодня"
}
Пример ответа:
{
  "answer": "мошенничество"
}
