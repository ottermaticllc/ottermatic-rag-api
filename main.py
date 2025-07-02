import os
import logging
import traceback
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client, Client
import uuid  # Add this import at the top
import re  # Add regex import for URL conversion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Ottermatic RAG Chatbot API",
    description="AI-powered chatbot with knowledge base search and conversational memory",
    version="1.0.6"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
)

# Models
class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    max_results: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    conversation_id: str
    user_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

# Auth
def verify_api_key(x_api_key: str = Header(None, alias="x-api-key")):
    expected_key = os.getenv('API_KEY')
    if not expected_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key

# Embedding
def create_embedding(text: str) -> List[float]:
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding error")

# Search
def search_knowledge_base(query_embedding: List[float], max_results: int = 5) -> List[dict]:
    try:
        result = supabase.rpc('match_documents', {
            'query_embedding': str(query_embedding),
            'match_threshold': 0.5,
            'match_count': max_results
        }).execute()
        return result.data or []
    except Exception as e:
        logger.warning(f"RPC search failed: {e}")
        try:
            result = supabase.table('knowledge_base').select('*').execute()
            return result.data[:max_results] if result.data else []
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

# Format context
def format_context(search_results: List[dict]) -> tuple[str, List[str], float]:
    if not search_results:
        return "No matching documents found.", [], 0.0
    context, sources, total_similarity = [], [], 0
    for i, result in enumerate(search_results):
        title = result.get("title", "Unknown")
        category = result.get("category", "General")
        content = result.get("content", "")
        sim = result.get("similarity", 0.5)
        context.append(f"Source {i+1} - {title} ({category}):\n{content}\n")
        sources.append(f"{title} ({category})")
        total_similarity += sim
    return "\n".join(context), sources, total_similarity / len(search_results)

# Memory
def save_conversation_message(user_id: str, conversation_id: str, role: str, message: str):
    try:
        supabase.table('conversation_logs').insert({
            'user_id': user_id,
            'conversation_id': conversation_id,
            'role': role,
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        logger.warning(f"Save message failed: {e}")

def get_conversation_context(conversation_id: str, max_messages: int = 10) -> List[dict]:
    try:
        result = supabase.table('conversation_logs').select('role, message, timestamp').eq(
            'conversation_id', conversation_id).order('timestamp', desc=False).limit(max_messages).execute()
        return result.data if result.data else []
    except Exception as e:
        logger.error(f"Load memory failed: {e}")
        return []

# Helper function to convert URLs to HTML links
def convert_urls_to_links(text: str) -> str:
    """Convert URLs in text to HTML anchor tags"""
    # Pattern to match URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    
    def replace_url(match):
        url = match.group(0)
        # Special handling for Calendly links
        if 'calendly.com' in url:
            return f'<a href="{url}" target="_blank" rel="noopener noreferrer">Book your 15-minute intro call here</a>'
        else:
            return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>'
    
    return re.sub(url_pattern, replace_url, text)

# Generate response
def generate_response_with_memory(question: str, rag_context: str, conversation_history: List[dict]) -> str:
    try:
        user_message_count = sum(1 for msg in conversation_history if msg['role'] == 'user')

        # Try multiple paths to find the prompt file
        possible_paths = [
            "prompts/otto_prompt.txt",
            "/var/task/prompts/otto_prompt.txt",
            os.path.join(os.path.dirname(__file__), "prompts", "otto_prompt.txt")
        ]

        # Add debug logging
        logger.info("🔍 Looking for prompt file...")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Script location: {__file__}")
        logger.info(f"Directory contents: {os.listdir('.')}")
        
        # Check if prompts directory exists
        if os.path.exists('prompts'):
            logger.info(f"Prompts directory contents: {os.listdir('prompts')}")

        prompt_template = None
        for path in possible_paths:
            logger.info(f"Trying path: {path}")
            try:
                with open(path, "r") as f:
                    prompt_template = f.read()
                    logger.info(f"✅ Loaded prompt from: {path}")
                    break
            except FileNotFoundError:
                logger.info(f"❌ Not found at: {path}")
                continue

        if not prompt_template:
            logger.error("❌ Could not find otto_prompt.txt in any expected location")
            raise FileNotFoundError("Prompt file not found")

        system_prompt = prompt_template.format(
            rag_context=rag_context,
            user_message_count=user_message_count
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": m['role'], "content": m['message']} for m in conversation_history[-10:])
        messages.append({"role": "user", "content": question})

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=400,
            temperature=0.6
        )

        answer = response.choices[0].message.content.strip()
        
        # Convert URLs to HTML links
        answer = convert_urls_to_links(answer)
        
        return answer

    except Exception as e:
        logger.error(f"❌ Generate response failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat generation error: {str(e)}")

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="running", message="Chatbot online v1.0.6")

@app.get("/health", response_model=HealthResponse)
async def health(api_key: str = Depends(verify_api_key)):
    try:
        openai_client.models.list()
        supabase.table("knowledge_base").select("id").limit(1).execute()
        return HealthResponse(status="healthy", message="Services operational")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service error")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    conversation_id = request.conversation_id or str(uuid.uuid4())
    history = get_conversation_context(conversation_id, max_messages=15)

    embedding = create_embedding(request.question)
    search_results = search_knowledge_base(embedding, request.max_results)
    rag_context, sources, confidence = format_context(search_results)
    answer = generate_response_with_memory(request.question, rag_context, history)

    if request.user_id:
        save_conversation_message(request.user_id, conversation_id, "user", request.question)
        save_conversation_message(request.user_id, conversation_id, "assistant", answer)

    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence=round(confidence, 3),
        conversation_id=conversation_id,
        user_id=request.user_id
    )
