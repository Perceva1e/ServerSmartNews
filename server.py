from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import feedparser
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class GlobalRecommender:
    def __init__(self):
        logger.info("Loading BERT model and tokenizer... (this may take 1-2 minutes)")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.bert_model.eval()
        if torch.cuda.is_available():
            self.bert_model = self.bert_model.cuda()
            logger.info("BERT moved to GPU")
        else:
            logger.info("BERT running on CPU")

        self.num_categories = 7
        self.categories = ['general', 'business', 'entertainment', 'health', 'science', 'sports', 'technology']
        self.moods = ['happy', 'sad', 'neutral']
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.mood_to_idx = {mood: idx for idx, mood in enumerate(self.moods)}

        self.rss_feeds = {
            'general': 'https://feeds.bbci.co.uk/news/rss.xml',
            'business': 'https://feeds.bbci.co.uk/news/business/rss.xml',
            'entertainment': 'https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml',
            'health': 'https://feeds.bbci.co.uk/news/health/rss.xml',
            'science': 'https://feeds.bbci.co.uk/news/science_and_environment/rss.xml',
            'sports': 'https://feeds.bbci.co.uk/sport/rss.xml',
            'technology': 'https://feeds.bbci.co.uk/news/technology/rss.xml'
        }
        logger.info("Global recommender initialized")

    def preprocess_text(self, text):
        if not text:
            return ""
        text = re.sub(r'\W', ' ', text.lower())
        return ' '.join(text.split())

    def extract_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def extract_features(self, news_list):
        embeddings, categories, moods = [], [], []
        for news in news_list:
            title = news.title or ''
            desc = news.description or ''
            cont = news.content or ''
            full_text = self.preprocess_text(title + ' ' + desc + ' ' + cont)
            embedding = self.extract_bert_embedding(full_text)
            embeddings.append(embedding)
            cat = news.category or 'general'
            categories.append(cat)
            mood = self.analyze_mood(full_text)
            moods.append(mood)

        embeddings = np.array(embeddings)

        cat_onehot = np.zeros((len(news_list), self.num_categories))
        mood_onehot = np.zeros((len(news_list), len(self.moods)))
        for i, (cat, mood) in enumerate(zip(categories, moods)):
            if cat in self.category_to_idx:
                cat_onehot[i, self.category_to_idx[cat]] = 1
            if mood in self.mood_to_idx:
                mood_onehot[i, self.mood_to_idx[mood]] = 1

        return np.hstack([embeddings, cat_onehot, mood_onehot])

    def analyze_mood(self, text):
        positive_keywords = {"happy", "joy", "success", "growth", "love", "peace", "hope", "celebration", "beautiful",
                             "good news"}
        negative_keywords = {"sad", "grief", "fear", "crisis", "death", "loss", "war", "failure", "decline", "negative",
                             "bad"}
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
        self.user_profile = features.mean(axis=0)

    def fetch_candidates_from_rss(self):
        candidate_news_list = []
        for category, rss_url in self.rss_feeds.items():
            try:
                feed = feedparser.parse(rss_url)
                for entry in feed.entries[:20]:
                    candidate_news_list.append(NewsItem(
                        title=entry.get('title'),
                        description=entry.get('summary'),
                        category=category,
                        url=entry.get('link'),
                        publishedAt=entry.get('published'),
                        urlToImage=entry.get('media_content', [{}])[0].get('url') if 'media_content' in entry else None,
                        content=entry.get('content', [{}])[0].get('value') if 'content' in entry else None
                    ))
            except Exception as e:
                logger.error(f"Error fetching RSS for {category}: {e}")
        return candidate_news_list

    def recommend_news(self, candidate_news_list, top_k=10):
        if self.user_profile is None:
            raise ValueError("Build user profile first.")

        cand_features = self.extract_features(candidate_news_list)
        cand_features = (cand_features - cand_features.mean(axis=0)) / (cand_features.std(axis=0) + 1e-8)

        similarities = cosine_similarity([self.user_profile], cand_features)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [candidate_news_list[i] for i in top_indices]

global_recommender = GlobalRecommender()

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    try:
        global_recommender.build_user_profile(request.saved_news)
        candidate_news_list = global_recommender.fetch_candidates_from_rss()

        if not candidate_news_list:
            raise HTTPException(status_code=500, detail="No candidate news fetched from RSS feeds")

        recommendations = global_recommender.recommend_news(candidate_news_list, top_k=20)
        return [rec.dict() for rec in recommendations]
    except Exception as e:
        logger.error(f"Error in /recommend: {e}")
        raise HTTPException(status_code=500, detail=str(e))