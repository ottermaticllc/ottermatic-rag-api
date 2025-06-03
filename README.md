# RAG Chatbot API

AI-powered chatbot with knowledge base search and conversational memory.

## Features

- ✅ **Vector Search**: Semantic search through knowledge base using OpenAI embeddings
- ✅ **Conversational Memory**: Maintains context across conversations
- ✅ **API Key Authentication**: Secure access control
- ✅ **FastAPI**: Auto-generated OpenAPI docs at `/docs`
- ✅ **Supabase Integration**: Vector database with pgvector
- ✅ **Make.com Ready**: Structured responses for automation

## API Endpoints

### Main Chat Endpoint
```
POST /api/chat
```

**Headers:**
```
x-api-key: YOUR_API_KEY
Content-Type: application/json
```

**Request Body:**
```json
{
  "question": "What is your refund policy?",
  "user_id": "user123",
  "conversation_id": "conv456",
  "max_results": 5
}
```

**Response:**
```json
{
  "answer": "Our refund policy allows...",
  "sources": ["Refund Policy (Support)", "Terms of Service (Legal)"],
  "confidence": 0.87,
  "conversation_id": "conv456",
  "user_id": "user123"
}
```

### Other Endpoints
- `GET /` - API status
- `GET /api/test` - Test configuration (no auth)
- `GET /health` - Health check (requires auth)
- `GET /docs` - Interactive API documentation

## Environment Variables

Set these in Vercel:

```
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://...
SUPABASE_KEY=eyJ...
API_KEY=your-secret-api-key
```

## Make.com Integration

Use the `/api/chat` endpoint with:
- **Method**: POST
- **Headers**: `x-api-key: YOUR_API_KEY`
- **Body**: JSON with `question`, `user_id`, `conversation_id`

## Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for interactive testing.

## Deployment

1. Push to GitHub
2. Connect to Vercel
3. Add environment variables
4. Deploy!

## Database Schema

Requires Supabase tables:
- `knowledge_base` - Documents with vector embeddings
- `conversation_logs` - Chat history
- `match_documents()` - Vector search function
