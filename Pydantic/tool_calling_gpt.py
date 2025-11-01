# ==============================================================================
# 🤖 AI-DRIVEN SUPPORT WORKFLOW (Pydantic + OpenAI + Instructor)
# ==============================================================================
# This tutorial-style script demonstrates a structured AI workflow using
# Pydantic, OpenAI, and the Instructor library.
#
# It consists of three automated stages:
#   1. Input Classification → enrich raw user queries with structure.
#   2. Tool Decision + Execution → decide which tools (functions) to call.
#   3. Structured Output → generate a final SupportTicket schema.
#
# Each stage is designed for explainability, schema validation, and safe
# interoperability with LLMs in real-world customer support systems.
# ==============================================================================


# ==============================================================================
# 1️⃣  IMPORTS AND ENVIRONMENT SETUP
# ==============================================================================
# Explanation:
# - pydantic: Defines strict data models for input/output validation.
# - pydantic_ai.Agent: A thin wrapper to structure AI outputs into Pydantic objects.
# - typing: Provides Literal, Optional, and List for type hints.
# - openai.OpenAI: Official OpenAI API client for direct model access.
# - instructor: A library that extends OpenAI responses to match Pydantic models.
# - dotenv: Loads environment variables (like OPENAI_API_KEY) from .env files.
# - nest_asyncio: Allows running async event loops inside Jupyter or notebooks.
# ==============================================================================

from pydantic import BaseModel, Field, EmailStr, field_validator
from pydantic_ai import Agent
from typing import Literal, Optional, List
from datetime import date, datetime
import json
from openai import OpenAI
import instructor
from dotenv import load_dotenv
import nest_asyncio

# Load .env variables (must include your OPENAI_API_KEY)
load_dotenv()

# Enables async event loops to run inside interactive shells like Jupyter
nest_asyncio.apply()


# ==============================================================================
# 2️⃣  SIMULATED DATABASES (Mock Backends)
# ==============================================================================
# Explanation:
# Instead of querying live systems, this script uses dictionaries/lists to
# emulate real database records. In a production system, these would be
# replaced by SQL queries, API calls, or CRM connectors.
# ==============================================================================

faq_db = [
    {
        "question": "How can I reset my password?",
        "answer": "Click 'Forgot Password' on the sign-in page and follow the email instructions.",
        "keywords": ["password", "reset", "account"]
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 3–5 business days. You can track your order via your account dashboard.",
        "keywords": ["shipping", "delivery", "order", "tracking"]
    },
    {
        "question": "How can I return an item?",
        "answer": "You can return any item within 30 days of purchase via the Returns page.",
        "keywords": ["return", "refund", "exchange"]
    },
]

# Each order record stores minimal metadata for lookups
order_db = {
    "ABC-12345": {
        "status": "shipped", "estimated_delivery": "2025-12-05",
        "purchase_date": "2025-12-01", "email": "joe@example.com"
    },
    "XYZ-23456": {
        "status": "processing", "estimated_delivery": "2025-12-15",
        "purchase_date": "2025-12-10", "email": "sue@example.com"
    },
    "QWE-34567": {
        "status": "delivered", "estimated_delivery": "2025-12-20",
        "purchase_date": "2025-12-18", "email": "bob@example.com"
    }
}


# ==============================================================================
# 3️⃣  DATA MODEL DEFINITIONS (Pydantic Schemas)
# ==============================================================================
# Explanation:
# These schemas define strict structures for data at each workflow stage.
# - UserInput: Validates the user's original message.
# - CustomerQuery: Adds semantic metadata such as priority and category.
# - FAQLookupArgs & CheckOrderStatusArgs: Define parameters for each tool call.
# - OrderDetails & SupportTicket: Represent the structured, final output.
# ==============================================================================

class UserInput(BaseModel):
    """Raw user query model with strict field validation."""
    name: str = Field(..., description="User's full name.")
    email: EmailStr = Field(..., description="Validated user email address.")
    query: str = Field(..., description="Text message or inquiry from the user.")
    order_id: Optional[str] = Field(None, description="Order ID if provided (format: ABC-12345).")
    purchase_date: Optional[date] = None

    # Custom validator for order_id using regex pattern
    @field_validator("order_id")
    def validate_order_id(cls, order_id):
        import re
        if order_id is None:
            return order_id
        pattern = r"^[A-Z]{3}-\d{5}$"
        if not re.match(pattern, order_id):
            raise ValueError("order_id must follow ABC-12345 (3 uppercase letters + dash + 5 digits)")
        return order_id


class CustomerQuery(UserInput):
    """Structured representation of enriched customer intent."""
    priority: Literal["low", "medium", "high"] = Field(..., description="Query urgency.")
    category: Literal["refund_request", "information_request", "other"] = Field(..., description="Categorized purpose of query.")
    is_complaint: bool = Field(..., description="True if this message expresses dissatisfaction.")
    tags: List[str] = Field(..., description="Keyword tags extracted from query context.")


class FAQLookupArgs(BaseModel):
    """Arguments expected by the FAQ lookup tool."""
    query: str
    tags: List[str]


class CheckOrderStatusArgs(BaseModel):
    """Arguments expected by the order status lookup tool."""
    order_id: str
    email: EmailStr


class OrderDetails(BaseModel):
    """Response schema for order lookups."""
    status: str
    estimated_delivery: Optional[str]
    note: str


class SupportTicket(CustomerQuery):
    """Final output schema integrating all stages of reasoning."""
    recommended_next_action: Literal[
        "escalate_to_agent", "send_faq_response", "send_order_status", "no_action_needed"
    ]
    order_details: Optional[OrderDetails] = None
    faq_response: Optional[str] = None
    creation_date: datetime = Field(default_factory=datetime.now)


# ==============================================================================
# 4️⃣  TOOL IMPLEMENTATIONS
# ==============================================================================
# Explanation:
# Tools are modular functions representing system capabilities. The AI can call
# them when appropriate. Each tool consumes a validated Pydantic schema to
# ensure safety and correctness.
# ==============================================================================

def lookup_faq_answer(args: FAQLookupArgs) -> str:
    """
    Search for an FAQ answer by comparing keywords and tags.
    Scoring is based on overlap between the query words and stored keywords.
    """
    query_words = set(word.lower() for word in args.query.split())
    tag_set = set(tag.lower() for tag in args.tags)
    best_match, best_score = None, 0

    for faq in faq_db:
        keywords = set(k.lower() for k in faq["keywords"])
        score = len(keywords & tag_set) + len(keywords & query_words)
        if score > best_score:
            best_score, best_match = score, faq

    return best_match["answer"] if best_match else "Sorry, no relevant FAQ found."


def check_order_status(args: CheckOrderStatusArgs) -> dict:
    """
    Retrieve the order status for a given order_id and email.
    Includes basic validation and descriptive notes.
    """
    order = order_db.get(args.order_id)
    if not order:
        return {"order_id": args.order_id, "status": "not found", "note": "Invalid order_id."}

    if args.email.lower() != order["email"].lower():
        return {"order_id": args.order_id, "status": order["status"],
                "estimated_delivery": order["estimated_delivery"],
                "note": "Email mismatch (order found for different customer)."}

    return {"order_id": args.order_id, "status": order["status"],
            "estimated_delivery": order["estimated_delivery"],
            "note": "Valid order and email match."}


# Register all available tool functions for dynamic lookup
available_tools = {
    "lookup_faq_answer": lookup_faq_answer,
    "check_order_status": check_order_status
}


# ==============================================================================
# 5️⃣  CORE WORKFLOW LOGIC
# ==============================================================================
# Explanation:
# Three main stages:
#   (1) classify_user_input → Structure the user query into CustomerQuery.
#   (2) decide_on_tools → Use OpenAI to decide which tools to invoke.
#   (3) generate_support_ticket → Produce structured SupportTicket schema.
# ==============================================================================

def classify_user_input(user_input: UserInput) -> CustomerQuery:
    """Stage 1: Convert freeform user input into structured intent fields."""
    print("\n🧩 Stage 1: Classifying user input...")
    agent = Agent(model="openai:gpt-4o", output_type=CustomerQuery)
    prompt = f"Analyze the following user input and create a structured CustomerQuery:\n{user_input.model_dump_json()}"
    response = agent.run_sync(prompt)
    print("✅ Classification complete.")
    return response.output


def decide_on_tools(customer_query: CustomerQuery, client: OpenAI):
    """Stage 2: Ask model which tool(s) to invoke given the structured intent."""
    print("\n🧠 Stage 2: Determining which tools to use...")
    tool_definitions = [
        {"type": "function", "function": {"name": "lookup_faq_answer", "description": "Look up a FAQ answer.", "parameters": FAQLookupArgs.model_json_schema()}},
        {"type": "function", "function": {"name": "check_order_status", "description": "Check a customer's order status.", "parameters": CheckOrderStatusArgs.model_json_schema()}}
    ]
    messages = [{"role": "user", "content": f"Decide which tools should be called for this query: {customer_query.model_dump_json()}"}]
    response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tool_definitions, tool_choice="auto")
    print("✅ Tool selection complete.")
    return response.choices[0].message


def generate_support_ticket(customer_query: CustomerQuery, tool_message, tool_outputs: list, client: instructor.Instructor) -> SupportTicket:
    """Stage 3: Combine user info, tool decisions, and results into a unified SupportTicket."""
    print("\n📄 Stage 3: Generating structured SupportTicket...")
    messages = [
        {"role": "user", "content": f"Initial structured query: {customer_query.model_dump_json()}"},
        tool_message.model_dump()  # Include assistant reasoning
    ]
    if tool_outputs:
        messages.extend(tool_outputs)

    # The Instructor client ensures OpenAI response matches the SupportTicket schema.
    result = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_model=SupportTicket
    )
    print("✅ SupportTicket successfully generated.")
    return result


# ==============================================================================
# 6️⃣  WORKFLOW EXECUTION (MAIN ORCHESTRATOR)
# ==============================================================================
# Explanation:
# This section wires all components together:
#   - Validates input
#   - Executes three stages sequentially
#   - Dynamically executes chosen tools
#   - Produces structured, human-readable SupportTicket output
# ==============================================================================

# Initialize both clients
openai_client = OpenAI()                       # Raw OpenAI interface
instructor_client = instructor.from_openai(openai_client)  # Schema-enforced interface

# Example user input (simulating live user message)
user_input_json = '''
{
    "name": "Joe User",
    "email": "joe@example.com",
    "query": "I'm really not happy with this product I bought",
    "order_id": "QWE-34567",
    "purchase_date": null
}
'''

# STEP 1: Validate JSON input into Pydantic model
validated_input = UserInput.model_validate_json(user_input_json)

# STEP 2: Stage 1 → classify input
customer_query = classify_user_input(validated_input)

# STEP 3: Stage 2 → decide on tools
tool_message = decide_on_tools(customer_query, openai_client)
tool_calls = getattr(tool_message, "tool_calls", None)
tool_outputs = []

# If the model decided to call tools, execute them safely
if tool_calls:
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_tools[function_name]

        # Validate tool arguments before invocation
        if function_name == "lookup_faq_answer":
            args = FAQLookupArgs.model_validate_json(tool_call.function.arguments)
        elif function_name == "check_order_status":
            args = CheckOrderStatusArgs.model_validate_json(tool_call.function.arguments)
        else:
            continue

        # Execute tool and capture output
        output = function_to_call(args)
        tool_outputs.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": json.dumps(output)
        })

# STEP 4: Stage 3 → create SupportTicket
support_ticket = generate_support_ticket(customer_query, tool_message, tool_outputs, instructor_client)

# STEP 5: Print final result in readable JSON format
print("\n🎯 FINAL STRUCTURED SUPPORT TICKET")
print(json.dumps(support_ticket.model_dump(), indent=2))