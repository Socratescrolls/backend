from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Create FastAPI app instance
app = FastAPI(
    title="Your Project API",
    description="Backend API for your application",
    version="0.1.0"
)

# CORS Configuration
origins = [
    "http://localhost:5173",  # Vite default dev server
    "http://127.0.0.1:5173",
    "http://localhost:3000",  # Common React dev server
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Example Request Model
class ItemRequest(BaseModel):
    name: str
    description: Optional[str] = None

# Example Response Model
class ItemResponse(ItemRequest):
    id: int

# Example Endpoint
@app.post("/items/", response_model=ItemResponse)
async def create_item(item: ItemRequest):
    """
    Create a new item
    
    - **name**: A required name for the item
    - **description**: An optional description
    """
    # Simulated item creation (replace with actual database logic)
    return ItemResponse(
        id=1,  # In real scenario, this would be a database-generated ID
        name=item.name,
        description=item.description
    )

@app.get("/items/", response_model=List[ItemResponse])
async def list_items():
    """
    Retrieve all items
    """
    # Simulated item retrieval (replace with actual database query)
    return [
        ItemResponse(id=1, name="Sample Item", description="A test item")
    ]

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}