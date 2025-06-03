from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
from openai import OpenAI
from supabase import create_client, Client
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ottermatic RAG Chatbot API",
    description="RAG-powered chatbot using vector embeddings from Supabase",
    version="1.0.0"
)

# Add CORS middleware for web requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')

if not supabase_key:
    raise ValueError("Either SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY must be set")

supabase: Client = create_client(supabase_url, supabase_key)

# Pydantic models for request/response
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

# API Key Authentication
def verify_api_key(x_api_key: str = Header(None, alias="x-api-key")):
    """Verify API key from header"""
    expected_key = os.getenv('API_KEY')
    if not expected_key:
        raise HTTPException(status_code=500, detail="API key not configured on server")
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required. Include 'x-api-key' header.")
    
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return x_api_key

# System prompt template
SYSTEM_PROMPT = """You are Ottermatic's AI assistant, a helpful and knowledgeable expert on business automation and AI integration for small businesses. Your goal is to answer client inquiries and try to point clients towards booking a discovery call within 5 messages unless user queries are irrelevant 

IMPORTANT INSTRUCTIONS:
- Answer questions using ONLY the provided context about Ottermatic's services and expertise
- If the answer isn't clearly in the provided context, say "I don't have specific information about that topic in my knowledge base, but I'd be happy to help you get in touch with our team for more details."
- Be conversational, helpful, and professional
- Always emphasize Ottermatic's focus on helping small businesses automate operations and leverage AI to save money and generate more revenue
- Include relevant service categories when appropriate
- If asked about pricing or specific implementation details, talk about customized pricing options and suggest meeting with a team member to discuss budget


CONTEXT FROM KNOWLEDGE BASE:
{context}

Based on the above context, please answer the following question in a helpful and conversational way:"""

def create_embedding(text: str) -> List[float]:
    """Create embedding for text using OpenAI API"""
    try:
        logger.info(f"Creating embedding for text: {text[:100]}...")
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        logger.info("✅ Successfully created embedding")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"❌ Failed to create embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create embedding: {str(e)}")

def search_knowledge_base(query_embedding: List[float], max_results: int = 5) -> List[dict]:
    """Search knowledge base using vector similarity"""
    try:
        logger.info(f"Searching knowledge base for {max_results} similar documents...")
        
        # Convert embedding to string format for Supabase
        embedding_str = str(query_embedding)
        
        # Perform vector similarity search using RPC function
        result = supabase.rpc(
            'match_documents',
            {
                'query_embedding': embedding_str,
                'match_threshold': 0.5,  # Minimum similarity threshold
                'match_count': max_results
            }
        ).execute()
        
        if result.data:
            logger.info(f"✅ Found {len(result.data)} matching documents via RPC function")
            return result.data
        else:
            # Fallback to basic similarity search if RPC function doesn't exist
            logger.warning("⚠️ RPC function not found, using basic similarity search")
            result = supabase.table('knowledge_base').select('*').execute()
            limited_results = result.data[:max_results] if result.data else []
            logger.info(f"✅ Found {len(limited_results)} documents via fallback method")
            return limited_results
            
    except Exception as e:
        logger.error(f"❌ Failed to search knowledge base: {e}")
        # Return empty list to continue with generic response
        return []

def format_context(search_results: List[dict]) -> tuple[str, List[str], float]:
    """Format search results into context string and extract sources"""
    if not search_results:
        logger.warning("⚠️ No search results to format")
        return "No specific information found in knowledge base.", [], 0.0
    
    context_parts = []
    sources = []
    total_similarity = 0
    
    for i, result in enumerate(search_results):
        title = result.get('title', 'Unknown')
        category = result.get('category', 'General')
        content = result.get('content', '')
        similarity = result.get('similarity', 0.5)
        
        # Add to context
        context_parts.append(f"Source {i+1} - {title} ({category}):\n{content}\n")
        
        # Track sources and similarity
        sources.append(f"{title} ({category})")
        total_similarity += similarity
    
    # Calculate average confidence
    avg_confidence = total_similarity / len(search_results) if search_results else 0.0
    
    context = "\n".join(context_parts)
    logger.info(f"✅ Formatted context from {len(search_results)} sources with avg confidence: {avg_confidence:.3f}")
    return context, sources, avg_confidence

def save_conversation_message(user_id: str, conversation_id: str, role: str, message: str):
    """Save a message to the conversation_logs table"""
    try:
        logger.info(f"💾 Saving {role} message to conversation {conversation_id}")
        
        result = supabase.table('conversation_logs').insert({
            'user_id': user_id,
            'conversation_id': conversation_id,
            'role': role,
            'message': message,
            'timestamp': 'now()'
        }).execute()
        
        logger.info(f"✅ Message saved successfully")
        return result.data
        
    except Exception as e:
        logger.error(f"❌ Failed to save conversation message: {e}")
        # Don't raise exception - conversation should continue even if logging fails
        return None

def get_conversation_context(conversation_id: str, max_messages: int = 10) -> str:
    """Retrieve recent conversation history for context"""
    try:
        logger.info(f"📚 Retrieving conversation context for {conversation_id}")
        
        result = supabase.table('conversation_logs').select('role, message, timestamp').eq(
            'conversation_id', conversation_id
        ).order('timestamp', desc=False).limit(max_messages).execute()
        
        if not result.data:
            logger.info("No previous conversation history found")
            return ""
        
        # Format conversation history
        context_parts = []
        for msg in result.data:
            role = msg['role'].title()  # User/Assistant
            message = msg['message']
            context_parts.append(f"{role}: {message}")
        
        context = "\n".join(context_parts)
        logger.info(f"✅ Retrieved {len(result.data)} previous messages")
        return context
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve conversation context: {e}")
        return ""  # Return empty context if retrieval fails

def generate_response_with_memory(question: str, rag_context: str, conversation_context: str) -> str:
    """Generate response using OpenAI GPT with both RAG context and conversation memory"""
    try:
        logger.info("🧠 Generating response with conversation memory...")
        
        # Enhanced system prompt that includes conversation context
        enhanced_prompt = f"""You are a helpful AI assistant with access to a knowledge base and conversation history.

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{conversation_context}

Instructions:
1. Use the knowledge base context to provide accurate, specific information
2. Reference the conversation history to maintain context and continuity
3. If the user refers to something from earlier in the conversation, acknowledge it
4. Provide helpful, conversational responses that feel natural
5. If you don't know something, say so clearly
6. Keep responses concise but complete

Answer the following question naturally and helpfully:"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        logger.info(f"✅ Generated response with memory: {answer[:100]}...")
        return answer
        
    except Exception as e:
        logger.error(f"❌ Failed to generate response with memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Ottermatic RAG Chatbot API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def detailed_health_check(api_key: str = Depends(verify_api_key)):
    """Detailed health check with service validation - requires API key"""
    try:
        logger.info("Performing detailed health check...")
        
        # Test OpenAI connection
        openai_client.models.list()
        logger.info("✅ OpenAI connection successful")
        
        # Test Supabase connection
        supabase.table('knowledge_base').select('id').limit(1).execute()
        logger.info("✅ Supabase connection successful")
        
        return HealthResponse(
            status="healthy",
            message="All services are operational"
        )
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """Main chat endpoint for RAG-powered responses with conversational memory - requires API key"""
    try:
        logger.info(f"🔥 Processing chat request from user: {request.user_id}")
        logger.info(f"📝 Question: {request.question[:100]}...")
        
        # Validate input
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Generate conversation_id if not provided
        if not request.conversation_id:
            import uuid
            request.conversation_id = str(uuid.uuid4())
            logger.info(f"🆕 Generated new conversation_id: {request.conversation_id}")
        
        # Get conversation context for memory
        conversation_context = get_conversation_context(request.conversation_id, max_messages=10)
        
        # Create embedding for the question
        query_embedding = create_embedding(request.question)
        
        # Search knowledge base
        search_results = search_knowledge_base(query_embedding, request.max_results)
        
        # Format context and extract sources
        rag_context, sources, confidence = format_context(search_results)
        
        # Generate response using GPT with both RAG context and conversation memory
        answer = generate_response_with_memory(request.question, rag_context, conversation_context)
        
        # Save user message to conversation log
        if request.user_id:
            save_conversation_message(request.user_id, request.conversation_id, "user", request.question)
            # Save assistant response to conversation log
            save_conversation_message(request.user_id, request.conversation_id, "assistant", answer)
        
        # Return response
        response = ChatResponse(
            answer=answer,
            sources=sources,
            confidence=round(confidence, 3),
            conversation_id=request.conversation_id,
            user_id=request.user_id
        )
        
        logger.info(f"✅ Chat request completed successfully for conversation: {request.conversation_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is working - no auth required"""
    return {
        "message": "API is working!",
        "openai_configured": bool(os.getenv('OPENAI_API_KEY')),
        "supabase_configured": bool(os.getenv('SUPABASE_URL')),
        "api_key_configured": bool(os.getenv('API_KEY')),
        "timestamp": "2024-01-01T00:00:00Z"
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)aa
