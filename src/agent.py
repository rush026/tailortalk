"""
LangGraph ReAct agent for TailorTalk — a saree visual similarity assistant.

Uses GPT-4o-mini with function calling. The agent can:
  - Chat naturally about sarees, fabrics, and fashion
  - Recognize when a similarity search is requested
  - Call the find_similar_sarees tool with the image source
  - Present results conversationally
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.tool import find_similar_sarees

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are **TailorTalk** — a knowledgeable and friendly AI saree style assistant 
for Byrappa Silks, a premium Indian saree retailer.

Your capabilities:
1. **Visual Similarity Search**: When the user provides an image (upload or URL), use the 
   `find_similar_sarees` tool to find visually similar sarees from the catalog.
2. **Fashion Expertise**: You can discuss saree fabrics (silk, organza, cotton, pashmina, 
   banarasi, etc.), weaving techniques, color palettes, styling advice, and occasion 
   recommendations.
3. **Product Information**: Share details about matched products including pricing, 
   availability, and direct links.

Guidelines:
- When the user uploads an image or gives you an image path/URL, ALWAYS call the 
  `find_similar_sarees` tool. Do NOT try to describe the image yourself.
- Present search results in a warm, conversational tone. Highlight what makes each 
  match visually similar (colour family, pattern style, fabric type based on the name).
- If the user asks for more results, call the tool again with a higher top_k.
- Keep responses concise but informative. Use emoji sparingly for warmth (✨, 🎨, etc.).
- If asked about something outside sarees/fashion, politely redirect to your domain.
- Always mention prices in ₹ (Indian Rupees).
- When presenting results, mention both the discounted price and MRP if different.
- If a saree is out of stock, mention it but suggest it might be restocked or that 
  similar alternatives are available.

Remember: You are a style advisor, not just a search engine. Add value with your fashion 
knowledge when presenting results."""


def create_agent(model_name: str = "gpt-4o-mini", temperature: float = 0.3):
    """Create and return the TailorTalk agent."""
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = create_react_agent(
        model=llm,
        tools=[find_similar_sarees],
        prompt=SYSTEM_PROMPT,
    )

    return agent


# Singleton agent instance
_agent = None


def get_agent():
    """Get or create the singleton agent instance."""
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent


def chat(message: str, thread_id: str = "default") -> str:
    """
    Send a message to the agent and return its response.
    Used for testing outside Streamlit.
    """
    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )

    # Extract the last AI message
    ai_messages = [m for m in result["messages"] if m.type == "ai" and m.content]
    return ai_messages[-1].content if ai_messages else "I couldn't generate a response."


if __name__ == "__main__":
    # Quick test
    print("TailorTalk Agent Test")
    print("=" * 50)
    response = chat("Hello! Can you help me find sarees?")
    print(response)
