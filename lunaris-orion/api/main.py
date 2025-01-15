from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import torch
import logging
from prometheus_client import Counter, Histogram, Gauge
import time
import jwt
import hashlib
import uuid
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Métricas Prometheus
GENERATION_REQUESTS = Counter('pixel_art_generations_total', 'Total de requisições de geração')
GENERATION_LATENCY = Histogram('generation_latency_seconds', 'Latência da geração de imagens')
ACTIVE_USERS = Gauge('active_users', 'Usuários ativos no sistema')
PREMIUM_USERS = Gauge('premium_users', 'Usuários premium')

app = FastAPI(
    title="Lunaris Orion API",
    description="API para geração de pixel art usando IA",
    version="2.0.0"
)

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sistema de autenticação
security = HTTPBearer()

# Classes de modelo
class GenerationRequest(BaseModel):
    prompt: str
    resolution: str = "32x32"
    style: Optional[str] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    palette: Optional[List[Dict[str, int]]] = None

class GenerationResponse(BaseModel):
    image_url: str
    seed: int
    generation_time: float
    remaining_generations: Optional[int] = None

class KeyRequest(BaseModel):
    user_id: str
    discord_id: str

class KeyResponse(BaseModel):
    key: str
    expires_at: datetime
    subscription_status: str

class SubscriptionInfo(BaseModel):
    status: str
    plan: str
    generations_today: int
    generations_limit: int
    expires_at: Optional[datetime] = None

# Funções de banco de dados
def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host="localhost",
        cursor_factory=RealDictCursor
    )

# Funções de autenticação e autorização
def create_api_key(user_id: str, discord_id: str) -> KeyResponse:
    key = str(uuid.uuid4())
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    expires_at = datetime.now() + timedelta(days=30)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Verifica se já existe uma chave ativa
        cur.execute("""
            UPDATE api_keys 
            SET status = 'revoked' 
            WHERE user_id = %s AND status = 'active'
        """, (user_id,))
        
        # Insere nova chave
        cur.execute("""
            INSERT INTO api_keys (user_id, key_hash, status, expires_at)
            VALUES (%s, %s, 'active', %s)
            RETURNING id
        """, (user_id, key_hash, expires_at))
        
        conn.commit()
        
        # Verifica status da assinatura
        cur.execute("""
            SELECT tier FROM subscriptions 
            WHERE user_id = %s AND status = 'active'
        """, (user_id,))
        subscription = cur.fetchone()
        status = subscription['tier'] if subscription else 'free'
        
        return KeyResponse(
            key=key,
            expires_at=expires_at,
            subscription_status=status
        )
    
    finally:
        cur.close()
        conn.close()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        key = credentials.credentials
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Verifica chave
        cur.execute("""
            SELECT ak.user_id, ak.expires_at, s.tier as subscription_tier,
                   COALESCE(
                       (SELECT COUNT(*) FROM generations 
                        WHERE user_id = ak.user_id 
                        AND DATE(created_at) = CURRENT_DATE
                       ), 0
                   ) as generations_today
            FROM api_keys ak
            LEFT JOIN subscriptions s ON s.user_id = ak.user_id AND s.status = 'active'
            WHERE ak.key_hash = %s AND ak.status = 'active'
        """, (key_hash,))
        
        result = cur.fetchone()
        if not result:
            raise HTTPException(status_code=401, detail="Chave API inválida")
        
        if result['expires_at'] < datetime.now():
            raise HTTPException(status_code=401, detail="Chave API expirada")
        
        # Verifica limite de gerações para plano gratuito
        if (result['subscription_tier'] != 'premium' and 
            result['generations_today'] >= 50):
            raise HTTPException(
                status_code=429, 
                detail="Limite diário de gerações atingido"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na verificação do token: {str(e)}")
        raise HTTPException(status_code=401, detail="Erro na autenticação")
    finally:
        cur.close()
        conn.close()

# Rotas da API
@app.get("/")
async def root():
    return {
        "status": "online",
        "version": "2.0.0",
        "model": "Lunaris Orion"
    }

@app.get("/status")
async def check_status():
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "name": props.name,
                "total_memory": props.total_memory / (1024**3),  # GB
                "compute_capability": f"{props.major}.{props.minor}"
            })
    
    return {
        "status": "operational",
        "gpu_available": torch.cuda.is_available(),
        "gpu_info": gpu_info,
        "active_users": ACTIVE_USERS._value.get(),
        "premium_users": PREMIUM_USERS._value.get()
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_pixel_art(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(verify_token)
):
    try:
        start_time = time.time()
        GENERATION_REQUESTS.inc()
        
        # Registra geração no banco
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            # Cria registro do prompt
            cur.execute("""
                INSERT INTO prompts (user_id, prompt, negative_prompt)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (user_info['user_id'], request.prompt, request.negative_prompt))
            prompt_id = cur.fetchone()['id']
            
            # Cria registro da geração
            cur.execute("""
                INSERT INTO generations (user_id, prompt_id, status)
                VALUES (%s, %s, 'processing')
                RETURNING id
            """, (user_info['user_id'], prompt_id))
            generation_id = cur.fetchone()['id']
            
            conn.commit()
            
            # TODO: Implementar geração real da imagem
            # Placeholder para demonstração
            generation_time = time.time() - start_time
            image_url = f"https://storage.lunaris.ai/generations/{generation_id}.png"
            
            # Atualiza registro com resultado
            cur.execute("""
                UPDATE generations
                SET status = 'completed', image_url = %s, completed_at = NOW()
                WHERE id = %s
            """, (image_url, generation_id))
            
            conn.commit()
            
            # Registra métricas
            GENERATION_LATENCY.observe(generation_time)
            
            # Calcula gerações restantes
            remaining = None
            if user_info['subscription_tier'] != 'premium':
                remaining = 50 - (user_info['generations_today'] + 1)
            
            return GenerationResponse(
                image_url=image_url,
                seed=request.seed or int(time.time()),
                generation_time=generation_time,
                remaining_generations=remaining
            )
            
        finally:
            cur.close()
            conn.close()
            
    except Exception as e:
        logger.error(f"Erro na geração: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro na geração: {str(e)}"
        )

@app.post("/keys", response_model=KeyResponse)
async def create_key(request: KeyRequest):
    try:
        return create_api_key(request.user_id, request.discord_id)
    except Exception as e:
        logger.error(f"Erro na criação da chave: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erro ao criar chave API"
        )

@app.get("/subscription/status", response_model=SubscriptionInfo)
async def get_subscription_status(user_info: dict = Depends(verify_token)):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Obtém informações da assinatura
        cur.execute("""
            SELECT tier, expires_at FROM subscriptions
            WHERE user_id = %s AND status = 'active'
        """, (user_info['user_id'],))
        subscription = cur.fetchone()
        
        # Obtém contagem de gerações do dia
        cur.execute("""
            SELECT COUNT(*) as count FROM generations
            WHERE user_id = %s AND DATE(created_at) = CURRENT_DATE
        """, (user_info['user_id'],))
        generations = cur.fetchone()
        
        return SubscriptionInfo(
            status="active",
            plan=subscription['tier'] if subscription else "free",
            generations_today=generations['count'],
            generations_limit=float('inf') if subscription and subscription['tier'] == 'premium' else 50,
            expires_at=subscription['expires_at'] if subscription else None
        )
        
    except Exception as e:
        logger.error(f"Erro ao obter status da assinatura: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erro ao obter informações da assinatura"
        )
    finally:
        cur.close()
        conn.close()

@app.get("/metrics")
async def metrics():
    return {
        "total_generations": GENERATION_REQUESTS._value.get(),
        "average_latency": GENERATION_LATENCY._sum.get() / GENERATION_LATENCY._count.get() if GENERATION_LATENCY._count.get() > 0 else 0,
        "active_users": ACTIVE_USERS._value.get(),
        "premium_users": PREMIUM_USERS._value.get()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 