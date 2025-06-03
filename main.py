from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
from datetime import datetime
from openai import OpenAI
from supabase import create_client, Client
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ottermatic RAG Chatbot API",
    description="AI-powered chatbot with knowledge base search and conversational memory",
    version="1.0.4"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
try:
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    supabase: Client = create_client(
        os.getenv('SUPABASE_URL'),
        os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    )
    logger.info("✅ Successfully initialized OpenAI and Supabase clients")
except Exception as e:
    logger.error(f"❌ Failed to initialize clients: {e}")

# Pydantic models
class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    max_results: Optional[int] = 5  # Number of knowledge base documents to retrieve

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    conversation_id: str
    user_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

# API Key authentication
def verify_api_key(x_api_key: str = Header(None, alias="x-api-key")):
    """Verify API key for secure access"""
    expected_key = os.getenv('API_KEY')
    if not expected_key:
        raise HTTPException(status_code=500, detail="API key not configured on server")
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key

def create_embedding(text: str) -> List[float]:
    """Generate vector embedding for text using OpenAI"""
    try:
        logger.info("🔍 Generating embedding for vector search...")
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        logger.info("✅ Embedding generated successfully")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create embedding")

def search_knowledge_base(query_embedding: List[float], max_results: int = 5) -> List[dict]:
    """Search Supabase knowledge base using vector similarity"""
    try:
        logger.info(f"🔎 Searching knowledge base for top {max_results} matches...")
        
        # Convert embedding to string format for Supabase RPC
        embedding_str = str(query_embedding)
        
        # Try using the match_documents RPC function first
        result = supabase.rpc(
            'match_documents',
            {
                'query_embedding': embedding_str,
                'match_threshold': 0.5,
                'match_count': max_results
            }
        ).execute()
        
        logger.info(f"📊 Found {len(result.data) if result.data else 0} relevant documents")
        return result.data or []
        
    except Exception as e:
        logger.warning(f"⚠️ RPC search failed, trying fallback method: {e}")
        
        # Fallback: simple table query
        try:
            result = supabase.table('knowledge_base').select('*').execute()
            fallback_results = result.data[:max_results] if result.data else []
            logger.info(f"📋 Fallback search returned {len(fallback_results)} documents")
            return fallback_results
        except Exception as fallback_error:
            logger.error(f"❌ All search methods failed: {fallback_error}")
            return []

def format_context(search_results: List[dict]) -> tuple[str, List[str], float]:
    """Format search results into context for LLM"""
    if not search_results:
        logger.warning("⚠️ No search results to format")
        return "No matching knowledge base entries found.", [], 0.0

    context_parts = []
    sources = []
    total_similarity = 0

    for i, result in enumerate(search_results):
        title = result.get("title", "Unknown Document")
        category = result.get("category", "General")
        content = result.get("content", "")
        similarity = result.get("similarity", 0.5)  # Default similarity if not provided
        
        context_parts.append(f"Source {i+1} - {title} ({category}):\n{content}\n")
        sources.append(f"{title} ({category})")
        total_similarity += similarity

    context = "\n".join(context_parts)
    avg_confidence = total_similarity / len(search_results)
    
    logger.info(f"📝 Formatted context from {len(search_results)} sources with avg confidence: {avg_confidence:.3f}")
    return context, sources, avg_confidence

def save_conversation_message(user_id: str, conversation_id: str, role: str, message: str):
    """Save individual message to conversation log"""
    try:
        result = supabase.table('conversation_logs').insert({
            'user_id': user_id,
            'conversation_id': conversation_id,
            'role': role,
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }).execute()
        logger.info(f"💾 Saved {role} message to conversation {conversation_id}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save conversation message: {e}")

def get_conversation_context(conversation_id: str, max_messages: int = 10) -> List[dict]:
    """Get recent conversation history as structured messages"""
    try:
        logger.info(f"📚 Loading conversation history for: {conversation_id}")
        
        result = supabase.table('conversation_logs').select(
            'role, message, timestamp'
        ).eq('conversation_id', conversation_id).order(
            'timestamp', desc=False  # Oldest first for proper conversation flow
        ).limit(max_messages).execute()
        
        messages = result.data if result.data else []
        logger.info(f"📖 Retrieved {len(messages)} previous messages for context")
        
        # Debug: Log the messages being retrieved
        for i, msg in enumerate(messages):
            logger.info(f"  {i+1}. {msg['role']}: {msg['message'][:50]}...")
            
        return messages
    except Exception as e:
        logger.error(f"❌ Error loading conversation memory: {e}")
        return []

def generate_response_with_memory(question: str, rag_context: str, conversation_history: List[dict]) -> str:
    """Generate response using both RAG context and conversation memory - OPTIMIZED"""
    try:
        # Build system prompt with RAG context
        system_content = f"""You are Ottermatic's helpful AI assistant with access to a comprehensive knowledge base.

KNOWLEDGE BASE CONTEXT:
{rag_context}

IMPORTANT INSTRUCTIONS:
- You have access to the conversation history and should reference it when answering
- If asked about previous questions or conversations, refer to the chat history
- Build on previous responses naturally 
- Use the knowledge base for factual information about Ottermatic
- If you don't know something from either source, say so clearly
- Keep responses helpful, professional, and conversational"""

        # Start with system message
        messages = [{"role": "system", "content": system_content}]
        
        # Add conversation history (maintain conversation flow)
        if conversation_history:
            logger.info(f"💭 Adding {len(conversation_history)} previous messages to context")
            for msg in conversation_history[-10:]:  # Last 10 messages for better context
                messages.append({
                    "role": msg['role'], 
                    "content": msg['message']
                })
        else:
            logger.info("🆕 No previous conversation history found")
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Debug: Log the conversation structure being sent
        logger.info(f"🔄 Sending {len(messages)} messages to OpenAI (gpt-4o-mini):")
        for i, msg in enumerate(messages):
            role = msg['role']
            content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            logger.info(f"  {i+1}. {role}: {content_preview}")
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",    # ✅ OPTIMIZED: 67x cheaper than gpt-4!
            messages=messages,
            max_tokens=1000,        # ✅ OPTIMIZED: Increased from 500 for fuller responses
            temperature=0.5         # ✅ OPTIMIZED: Balanced creativity vs consistency
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info("✅ Response generated with gpt-4o-mini (fast & cost-effective)")
        return answer
        
    except Exception as e:
        logger.error(f"❌ Chat generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API status check"""
    return HealthResponse(
        status="running",
        message="RAG Chatbot API (Optimized)"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check(api_key: str = Depends(verify_api_key)):
    """Detailed health check - requires API key"""
    try:
        # Test OpenAI connection
        openai_client.models.list()
        
        # Test Supabase connection
        supabase.table("knowledge_base").select("id").limit(1).execute()
        
        logger.info("✅ Health check passed - all systems operational")
        return HealthResponse(
            status="healthy",
            message="All systems operational (gpt-4o-mini optimized)"
        )
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """Main chat endpoint with RAG and conversation memory - OPTIMIZED"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    import uuid
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    logger.info(f"🔄 Processing chat request for conversation: {conversation_id}")
    logger.info(f"❓ User question: {request.question}")

    try:
        # FIRST: Get conversation history BEFORE generating response
        conversation_history = get_conversation_context(conversation_id, max_messages=15)
        
        # Create embedding and search knowledge base
        logger.info("🔍 Creating embedding for knowledge base search...")
        embedding = create_embedding(request.question)
        search_results = search_knowledge_base(embedding, request.max_results)
        rag_context, sources, confidence = format_context(search_results)
        
        # Generate response using both RAG and conversation memory
        logger.info("🤖 Generating response with optimized gpt-4o-mini...")
        answer = generate_response_with_memory(request.question, rag_context, conversation_history)

        # THEN: Save the NEW conversation messages AFTER generating response
        if request.user_id:
            logger.info("💾 Saving NEW conversation messages...")
            save_conversation_message(request.user_id, conversation_id, "user", request.question)
            save_conversation_message(request.user_id, conversation_id, "assistant", answer)
        
        logger.info(f"✅ Chat response completed - Confidence: {confidence:.3f}")
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence=round(confidence, 3),
            conversation_id=conversation_id,
            user_id=request.user_id
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"❌ Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify configuration (no auth required)"""
    return {
        "message": "API is working! (Optimized with gpt-4o-mini)",
        "model": "gpt-4o-mini",
        "max_tokens": 1000,
        "temperature": 0.5,
        "openai_configured": bool(os.getenv('OPENAI_API_KEY')),
        "supabase_configured": bool(os.getenv('SUPABASE_URL')),
        "api_key_configured": bool(os.getenv('API_KEY')),
        "timestamp": datetime.utcnow().isoformat()
    }

# Local development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
