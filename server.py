from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI()

class NewsItem(BaseModel):
    title: str | None = None
    description: str | None = None
    category: str = "general"
    url: str | None = None
    publishedAt: str | None = None
    urlToImage: str | None = None
    content: str | None = None

class RecommendRequest(BaseModel):
    saved_news: list[NewsItem]

class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class NewsRecommender:
    def __init__(self, num_categories=7, embedding_dim=32, num_epochs=50, lr=0.001):
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.autoencoder = None
        self.user_profile = None
        self.categories = ['general', 'business', 'entertainment', 'health', 'science', 'sports', 'technology']
        self.moods = ['happy', 'sad', 'neutral']
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.mood_to_idx = {mood: idx for idx, mood in enumerate(self.moods)}

    def preprocess_text(self, text):
        if not text:
            return ""
        text = re.sub(r'\W', ' ', text.lower())
        return ' '.join(text.split())

    def extract_features(self, news_list, fit_vectorizer=True):
        texts, categories, moods = [], [], []
        for news in news_list:
            title = news.title or ''
            desc = news.description or ''
            full_text = self.preprocess_text(title + ' ' + desc)
            texts.append(full_text)
            cat = news.category or 'general'
            categories.append(cat)
            mood = self.analyze_mood(full_text)
            moods.append(mood)

        if fit_vectorizer:
            tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        else:
            tfidf_matrix = self.vectorizer.transform(texts).toarray()

        cat_onehot = np.zeros((len(news_list), self.num_categories))
        mood_onehot = np.zeros((len(news_list), len(self.moods)))
        for i, (cat, mood) in enumerate(zip(categories, moods)):
            if cat in self.category_to_idx:
                cat_onehot[i, self.category_to_idx[cat]] = 1
            if mood in self.mood_to_idx:
                mood_onehot[i, self.mood_to_idx[mood]] = 1

        return np.hstack([tfidf_matrix, cat_onehot, mood_onehot])

    def analyze_mood(self, text):
        positive_keywords = {"happy", "joy", "success", "growth", "love", "peace", "hope", "celebration", "beautiful", "good news"}
        negative_keywords = {"sad", "grief", "fear", "crisis", "death", "loss", "war", "failure", "decline", "negative", "bad"}
        text_lower = text.lower()
        if any(kw in text_lower for kw in positive_keywords):
            return "happy"
        elif any(kw in text_lower for kw in negative_keywords):
            return "sad"
        return "neutral"

    def build_user_profile(self, saved_news_list):
        if not saved_news_list:
            raise ValueError("No saved news provided.")

        features = self.extract_features(saved_news_list)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        dataset = TensorDataset(torch.FloatTensor(features))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        input_dim = features.shape[1]
        self.autoencoder = Autoencoder(input_dim, self.embedding_dim)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.lr)

        self.autoencoder.train()
        for _ in range(self.num_epochs):
            for batch_features, in dataloader:
                optimizer.zero_grad()
                reconstructed, _ = self.autoencoder(batch_features)
                loss = criterion(reconstructed, batch_features)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            _, latent = self.autoencoder(torch.FloatTensor(features))
            self.user_profile = latent.mean(dim=0).numpy()

    def recommend_news(self, candidate_news_list, top_k=10):
        if self.user_profile is None:
            raise ValueError("Build user profile first.")

        cand_features = self.extract_features(candidate_news_list, fit_vectorizer=False)
        cand_features = (cand_features - cand_features.mean(axis=0)) / (cand_features.std(axis=0) + 1e-8)

        with torch.no_grad():
            _, latent_cand = self.autoencoder(torch.FloatTensor(cand_features))
            latent_cand = latent_cand.numpy()

        similarities = cosine_similarity([self.user_profile], latent_cand)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [candidate_news_list[i] for i in top_indices]

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    try:
        recommender = NewsRecommender(num_epochs=20)
        recommender.build_user_profile(request.saved_news)

        NEWS_API_KEY = "4c0c19e6cad4422a8a177baf5a64ded3"
        candidate_news_list = []
        for category in recommender.categories:
            url = f"https://newsapi.org/v2/top-headlines?category={category}&apiKey={NEWS_API_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json()["articles"]
                for article in articles:
                    candidate_news_list.append(NewsItem(
                        title=article.get("title"),
                        description=article.get("description"),
                        category=category,
                        url=article.get("url"),
                        publishedAt=article.get("publishedAt"),
                        urlToImage=article.get("urlToImage"),
                        content=article.get("content")
                    ))

        if not candidate_news_list:
            raise HTTPException(status_code=500, detail="No candidate news fetched from NewsAPI")

        recommendations = recommender.recommend_news(candidate_news_list, top_k=20)
        return [rec.dict() for rec in recommendations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))