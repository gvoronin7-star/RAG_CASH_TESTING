"""
Модуль для работы с эмбеддингами и векторным хранилищем FAISS.

Здесь мы создаем векторные представления текстов используя OpenAI API
и сохраняем их в FAISS для быстрого семантического поиска.
"""

import faiss
import numpy as np
from openai import OpenAI
from typing import List, Tuple
import os
import pickle


class EmbeddingStore:
    """
    Класс для работы с векторным хранилищем FAISS.
    
    Использует OpenAI API для создания эмбеддингов
    и FAISS для их хранения и поиска.
    """
    
    # Размерность для модели text-embedding-3-small
    EMBEDDING_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self, 
        collection_name: str = "documents",
        persist_directory: str = "./faiss_db",
        embedding_model: str = "text-embedding-3-small",
        api_key: str = None,
        base_url: str = "https://openai.api.proxyapi.ru/v1"
    ):
        """
        Инициализация хранилища эмбеддингов.
        
        Args:
            collection_name: Имя коллекции (используется для именования файлов)
            persist_directory: Директория для сохранения данных FAISS
            embedding_model: Название модели OpenAI для эмбеддингов
            api_key: API ключ OpenAI (если None, берется из переменной окружения)
            base_url: URL прокси для OpenAI API
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Определяем размерность эмбеддингов
        self.embedding_dim = self.EMBEDDING_DIMENSIONS.get(
            embedding_model, 
            self.EMBEDDING_DIMENSIONS["text-embedding-3-small"]
        )
        
        print(f"FAISS init in directory: {persist_directory}")
        
        # Создаем директорию если не существует
        os.makedirs(persist_directory, exist_ok=True)
        
        # Инициализируем клиент OpenAI для создания эмбеддингов
        self.openai_client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
        
        print(f"Model: {embedding_model} (dimension: {self.embedding_dim})")
        
        # Загружаем или создаем индекс FAISS
        self._load_or_create_index()
        
        print(f"OK. FAISS initialized. Documents: {self.ntotal}")
    
    def _load_or_create_index(self) -> None:
        """Загружает существующий индекс или создает новый."""
        index_path = os.path.join(self.persist_directory, f"{self.collection_name}_index.faiss")
        metadata_path = os.path.join(self.persist_directory, f"{self.collection_name}_metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # Загружаем существующий индекс
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.documents = metadata.get('documents', [])
                self.sources = metadata.get('sources', [])
        else:
            # Создаем новый индекс
            # Используем IndexFlatIP для косинусного сходства (после нормализации)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.documents = []
            self.sources = []
    
    def _save_index(self) -> None:
        """Сохраняет индекс на диск."""
        index_path = os.path.join(self.persist_directory, f"{self.collection_name}_index.faiss")
        metadata_path = os.path.join(self.persist_directory, f"{self.collection_name}_metadata.pkl")
        
        faiss.write_index(self.index, index_path)
        
        metadata = {
            'documents': self.documents,
            'sources': self.sources
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    @property
    def ntotal(self) -> int:
        """Возвращает количество документов в индексе."""
        return self.index.ntotal
    
    def _create_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Разбивает текст на чанки (фрагменты) с перекрытием."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Создает эмбеддинги для списка текстов используя OpenAI API."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts,
                encoding_format="float"
            )
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            print(f"Error: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Tuple[str, str]]) -> None:
        """Добавляет документы в векторное хранилище."""
        all_chunks = []
        all_sources = []
        
        print(f"\nAdding {len(documents)} documents to FAISS...")
        
        for doc_name, doc_text in documents:
            chunks = self._create_chunks(doc_text)
            print(f"  - {doc_name}: {len(chunks)} chunks")
            
            for chunk in chunks:
                all_chunks.append(chunk)
                all_sources.append(doc_name)
        
        print(f"\nCreating embeddings for {len(all_chunks)} chunks via OpenAI API...")
        
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            print(f"  Processing chunks {i+1}-{min(i+batch_size, len(all_chunks))} of {len(all_chunks)}...")
            
            batch_embeddings = self._create_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
        
        # Нормализуем векторы для косинусного сходства
        embeddings_array = np.array(all_embeddings, dtype='float32')
        faiss.normalize_L2(embeddings_array)
        
        # Добавляем в индекс
        self.index.add(embeddings_array)
        self.documents.extend(all_chunks)
        self.sources.extend(all_sources)
        
        # Сохраняем на диск
        self._save_index()
        
        print(f"OK. Added {len(all_chunks)} chunks. Total: {self.ntotal}")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Выполняет семантический поиск по векторному хранилищу."""
        if self.ntotal == 0:
            print("WARNING: Database is empty, no documents for search")
            return []
        
        # Создаем эмбеддинг для запроса
        query_embeddings = self._create_embeddings([query])
        query_vector = np.array([query_embeddings[0]], dtype='float32')
        faiss.normalize_L2(query_vector)
        
        # Выполняем поиск
        k = min(top_k, self.ntotal)
        distances, indices = self.index.search(query_vector, k)
        
        # Форматируем результаты
        formatted_results = []
        for i in range(k):
            idx = indices[0][i]
            if idx >= 0:
                chunk_text = self.documents[idx]
                source = self.sources[idx]
                distance = float(distances[0][i])
                formatted_results.append((chunk_text, source, distance))
        
        return formatted_results
    
    def clear_collection(self) -> None:
        """Очищает коллекцию (удаляет все документы)."""
        self.index.reset()
        self.documents = []
        self.sources = []
        self._save_index()
        print("OK. Collection cleared")


def get_sample_documents() -> List[Tuple[str, str]]:
    """Возвращает примеры документов для демонстрации RAG."""
    documents = [
        (
            "Python Основы",
            """
            Python - это высокоуровневый язык программирования общего назначения. 
            Он был создан Гвидо ван Россумом и впервые выпущен в 1991 году.
            
            Python известен своей простотой и читаемостью кода. Философия языка 
            подчеркивает важность читаемости кода и позволяет программистам 
            выражать концепции в меньшем количестве строк кода.
            
            Основные возможности Python включают:
            - Динамическую типизацию
            - Автоматическое управление памятью
            - Обширную стандартную библиотеку
            - Поддержку множественных парадигм программирования
            
            Python широко используется в веб-разработке, анализе данных, 
            машинном обучении, автоматизации и научных вычислениях.
            """
        ),
        (
            "Машинное обучение и AI",
            """
            Машинное обучение (Machine Learning) - это подраздел искусственного 
            интеллекта, который изучает алгоритмы и статистические модели, 
            позволяющие компьютерам выполнять задачи без явного программирования.
            
            Основные типы машинного обучения:
            
            1. Обучение с учителем (Supervised Learning)
            В этом подходе модель обучается на размеченных данных.
            
            2. Обучение без учителя (Unsupervised Learning)
            Модель ищет закономерности в неразмеченных данных.
            
            3. Обучение с подкреплением (Reinforcement Learning)
            Агент обучается принимать решения, взаимодействуя со средой.
            
            RAG (Retrieval-Augmented Generation) - это техника, которая улучшает 
            качество ответов языковых моделей, дополняя их внешними знаниями.
            """
        ),
        (
            "Векторные базы данных",
            """
            Векторные базы данных - это специализированные системы хранения данных, 
            оптимизированные для хранения и поиска векторных эмбеддингов.
            
            Что такое эмбеддинги?
            Эмбеддинги - это векторные представления данных в многомерном пространстве.
            Семантически похожие объекты располагаются близко друг к другу.
            
            FAISS (Facebook AI Similarity Search) - это библиотека для эффективного 
            поиска схожих векторов, разработанная Facebook Research.
            
            Преимущества FAISS:
            - Высокая скорость поиска
            - Поддержка различных алгоритмов индексации
            - Работа с большими объемами данных
            - Эффективное использование памяти
            
            Векторные базы данных критически важны для RAG-систем, так как они 
            позволяют быстро находить релевантные документы на основе семантического 
            сходства запроса с содержимым базы данных.
            
            OpenAI предоставляет мощные модели для создания эмбеддингов, такие как 
            text-embedding-3-small и text-embedding-3-large.
            """
        )
    ]
    
    return documents
