import openai
import os
from typing import List, Tuple, Dict
from dotenv import load_dotenv
import difflib
from functools import lru_cache
import hashlib

# --- Конфигурация ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Константы
SYSTEM_PROMPT = "Ты — ассистент СКК. Отвечай кратко. Если информации нет, скажи об этом."
DEFAULT_RESPONSE = "Извините, я не нашел ответа на ваш вопрос."
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.3
MAX_TOKENS = 150
SIMILARITY_GOOD = 0.6  # Порог хорошего совпадения
SIMILARITY_PARTIAL = 0.4  # Порог частичного совпадения
MAX_CONTEXT_LENGTH = 300
CACHE_SIZE = 256

# --- Вспомогательные функции ---
def clean_response(answer: str) -> str:
    """Удаляет нумерацию из начала ответа"""
    if answer and answer[0].isdigit() and '. ' in answer[:5]:
        return answer.split('. ', 1)[1]
    return answer

def get_text_hash(text: str) -> str:
    """Генерирует хэш для текста"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def compress_text(text: str, max_len: int = MAX_CONTEXT_LENGTH) -> str:
    """Сжимает текст до указанной длины"""
    if len(text) <= max_len:
        return text
    return text[:max_len//2] + " [...] " + text[-max_len//2:]

def should_process(question: str) -> bool:
    """Проверяет, стоит ли обрабатывать вопрос"""
    question = question.strip().lower()
    return len(question) > 3 and question not in {"спасибо", "привет", "здравствуйте"}

# --- Основная логика ---
@lru_cache(maxsize=CACHE_SIZE)
def load_qa_pairs(questions_path: str, answers_path: str) -> List[Tuple[str, str]]:
    """Загружает и кэширует пары вопрос-ответ"""
    try:
        with open(questions_path, 'r', encoding='utf-8') as f_q, \
             open(answers_path, 'r', encoding='utf-8') as f_a:
            questions = [q.strip() for q in f_q if q.strip()]
            answers = [clean_response(a.strip()) for a in f_a if a.strip()]
            return list(zip(questions, answers))
    except FileNotFoundError:
        return []

def find_best_match_extended(user_question: str, qa_pairs: List[Tuple[str, str]]) -> Tuple[str, str, float]:
    """Расширенный поиск с оценкой схожести"""
    if not qa_pairs:
        return "", "", 0.0
    
    best_match, best_answer, best_score = "", "", 0.0
    
    for question, answer in qa_pairs:
        current_score = difflib.SequenceMatcher(
            None, 
            user_question.lower().strip(), 
            question.lower().strip()
        ).ratio()
        
        if current_score > best_score:
            best_score = current_score
            best_match = question
            best_answer = answer
            
            # Оптимизация: прерываем если нашли идеальное совпадение
            if best_score >= 0.95:
                break
    
    return best_match, best_answer, best_score

@lru_cache(maxsize=CACHE_SIZE)
def get_api_response(messages: Tuple[Tuple[str, str]]) -> str:
    """Кэшированный запрос к API"""
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": r, "content": c} for r, c in messages],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            frequency_penalty=0.5
        )
        return clean_response(response.choices[0].message['content'])
    except Exception:
        return DEFAULT_RESPONSE

def generate_smart_response(user_question: str, matched_question: str, 
                          matched_answer: str, similarity_score: float) -> str:
    """Генерирует ответ с учетом степени схожести"""
    if similarity_score >= SIMILARITY_GOOD:
        return matched_answer
    elif similarity_score >= SIMILARITY_PARTIAL:
        # Для частичных совпадений добавляем контекст
        messages = (
            ("system", SYSTEM_PROMPT),
            ("user", f"Вопрос: {user_question}\n"
                    f"Похожий вопрос: {matched_question}\n"
                    f"Ответ на похожий вопрос: {matched_answer}\n\n"
                    f"Дайте краткий ответ на первый вопрос, используя второй ответ как контекст.")
        )
        response = get_api_response(messages)
        return f"Возможно, это подходящий ответ:\n\n{response}"
    else:
        return DEFAULT_RESPONSE

# --- Основной интерфейс ---
def ask_assistant(user_question: str, 
                 questions_file: str = "knowledge.txt", 
                 answers_file: str = "knowledge1.txt") -> str:
    """Улучшенная функция обработки вопросов"""
    if not user_question.strip():
        return "Пожалуйста, задайте вопрос."
    
    if not should_process(user_question):
        return DEFAULT_RESPONSE
    
    qa_pairs = load_qa_pairs(questions_file, answers_file)
    matched_question, matched_answer, similarity_score = find_best_match_extended(user_question, qa_pairs)
    
    if not matched_answer:
        return DEFAULT_RESPONSE
    
    return generate_smart_response(
        user_question=user_question,
        matched_question=matched_question,
        matched_answer=matched_answer,
        similarity_score=similarity_score
    )

# --- CLI Интерфейс ---
def main():
    """Точка входа для командной строки"""
    print("Ассистент СКК (для выхода нажмите Ctrl+C)")
    while True:
        try:
            question = input("\nВаш вопрос: ").strip()
            if not question:
                continue
                
            answer = ask_assistant(question)
            print("\nОтвет:", answer)
            
        except KeyboardInterrupt:
            print("\nЗавершение работы...")
            break
        except Exception as e:
            print(f"\nОшибка: {str(e)}")

if __name__ == "__main__":
    main()