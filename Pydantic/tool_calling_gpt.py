# ==============================================================================
# 1. IMPORTS
# ==============================================================================
# Explanation:
# This section imports the necessary libraries for your multi-stage workflow.
# - `pydantic`: The foundation for data validation and structuring.
# - `pydantic_ai`: A high-level library used here for the initial classification step (Stage 1).
# - `openai`: The official client library for interacting with the OpenAI API.
# - `instructor`: A library that patches the `openai` client to enforce Pydantic
#   schemas on the AI's output, crucial for the final structuring step (Stage 3).
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

load_dotenv()
nest_asyncio.apply()

# ==============================================================================
# 2. FAKE DATABASES
# ==============================================================================
# Explanation:
# These are mock databases. In a real application, your tools would query a
# live database or an external API. These dictionaries simulate that
# environment, allowing your tools to retrieve data and test the logic.
# ==============================================================================
faq_db = [
    {
        "question": "How can I reset my password?",
        "answer": "To reset your password, click 'Forgot Password' on the sign-in page and follow the instructions sent to your email.",
        "keywords": ["password", "reset", "account"]
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 3-5 business days. You can track your order in your account dashboard.",
        "keywords": ["shipping", "delivery", "order", "tracking"]
    },
    {
        "question": "How can I return an item?",
        "answer": "You can return any item within 30 days of purchase. Visit our returns page to start the process.",
        "keywords": ["return", "refund", "exchange"]
    },
]

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
# 3. PYDANTIC MODEL DEFINITIONS
# ==============================================================================
# Explanation:
# These models are the structural backbone of your application. Each class
# defines a specific data schema.
# - `UserInput`: Validates the raw input from the user.
# - `CustomerQuery`: Represents the enriched data after initial AI classification.
# - `FAQLookupArgs` & `CheckOrderStatusArgs`: Define the exact arguments your
#   tools require, preventing the AI from calling them with incorrect data.
# - `SupportTicket`: The final, comprehensive schema that the entire process
#   aims to produce.
# ==============================================================================
class UserInput(BaseModel):
    name: str = Field(..., description="User's name")
    email: EmailStr = Field(..., description="User's email address")
    query: str = Field(..., description="User's query")
    order_id: Optional[str] = Field(None, description="Order ID if available (format: ABC-12345)")

    @field_validator("order_id")
    def validate_order_id(cls, order_id):
        import re
        if order_id is None: return order_id
        pattern = r"^[A-Z]{3}-\d{5}$"
        if not re.match(pattern, order_id):
            raise ValueError("order_id must be in format ABC-12345 (3 uppercase letters, dash, 5 digits)")
        return order_id
    purchase_date: Optional[date] = None

class CustomerQuery(UserInput):
    priority: str = Field(..., description="Priority level: low, medium, or high")
    category: Literal['refund_request', 'information_request', 'other'] = Field(..., description="Query category")
    is_complaint: bool = Field(..., description="Whether this is a complaint")
    tags: List[str] = Field(..., description="Relevant keyword tags")

class FAQLookupArgs(BaseModel):
    query: str = Field(..., description="User's query")
    tags: List[str] = Field(..., description="Relevant keyword tags from the customer query")

class CheckOrderStatusArgs(BaseModel):
    order_id: str = Field(..., description="Customer's order ID (format: ABC-12345)")
    email: EmailStr = Field(..., description="Customer's email address")

class OrderDetails(BaseModel):
    status: str
    estimated_delivery: Optional[str]
    note: str

class SupportTicket(CustomerQuery):
    recommended_next_action: Literal['escalate_to_agent', 'send_faq_response', 'send_order_status', 'no_action_needed']
    order_details: Optional[OrderDetails] = None
    faq_response: Optional[str] = None
    creation_date: datetime = Field(default_factory=datetime.now)

# ==============================================================================
# 4. TOOL IMPLEMENTATIONS
# ==============================================================================
# Explanation:
# These functions are the "tools" that the AI can choose to call. They represent
# the actions your system can take, such as looking up data. The Pydantic
# argument models (`FAQLookupArgs`, `CheckOrderStatusArgs`) ensure that the AI
# provides valid input when it decides to use these tools.
# ==============================================================================
def lookup_faq_answer(args: FAQLookupArgs) -> str:
    """Look up an FAQ answer by matching tags and words in query to FAQ entry keywords."""
    query_words = set(word.lower() for word in args.query.split())
    tag_set = set(tag.lower() for tag in args.tags)
    best_match, best_score = None, 0
    for faq in faq_db:
        keywords = set(k.lower() for k in faq["keywords"])
        score = len(keywords & tag_set) + len(keywords & query_words)
        if score > best_score:
            best_score, best_match = score, faq
    return best_match["answer"] if best_match and best_score > 0 else "Sorry, I couldn't find an FAQ answer."

def check_order_status(args: CheckOrderStatusArgs) -> dict:
    """Checks the status of a customer's order by order_id and email."""
    order = order_db.get(args.order_id)
    if not order:
        return {"order_id": args.order_id, "status": "not found", "note": "order_id not found"}
    if args.email.lower() != order.get("email", "").lower():
        return {"order_id": args.order_id, "status": order["status"], "estimated_delivery": order["estimated_delivery"], "note": "order_id found but email mismatch"}
    return {"order_id": args.order_id, "status": order["status"], "estimated_delivery": order["estimated_delivery"], "note": "order_id and email match"}

available_tools = {"lookup_faq_answer": lookup_faq_answer, "check_order_status": check_order_status}

# ==============================================================================
# 5. CORE LOGIC (MULTI-STEP WORKFLOW)
# ==============================================================================
# Explanation:
# This section defines the three distinct stages of your AI workflow.
# Each function performs a specific AI call with a clear objective.
#===============================================================================

# STAGE 1: Classify the initial user input into a CustomerQuery object.
def classify_user_input(user_input: UserInput) -> CustomerQuery:
    """Uses pydantic_ai.Agent to enrich the raw user input."""
    print("Stage 1: Classifying user input...")
    agent = Agent(model="openai:gpt-4o", output_type=CustomerQuery)
    prompt = f"Analyze the following user data and generate a structured customer query object from it: {user_input.model_dump_json()}"
    response = agent.run_sync(prompt)
    print("Classification complete.")
    return response.output

# STAGE 2: Decide on tool use based on the classified query.
def decide_on_tools(customer_query: CustomerQuery, client: OpenAI):
    """Uses a standard OpenAI client to determine if any tools should be called."""
    print("\nStage 2: Deciding on tool use...")
    tool_definitions = [
        {"type": "function", "function": {"name": "lookup_faq_answer", "description": "Look up an FAQ answer.", "parameters": FAQLookupArgs.model_json_schema()}},
        {"type": "function", "function": {"name": "check_order_status", "description": "Check the status of a customer's order.", "parameters": CheckOrderStatusArgs.model_json_schema()}}
    ]
    messages = [{"role": "user", "content": f"Based on this query, decide which tools to call: {customer_query.model_dump_json()}"}]
    response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tool_definitions, tool_choice="auto")
    print("Tool decision complete.")
    return response.choices[0].message

# STAGE 3: Generate the final, structured SupportTicket.
def generate_support_ticket(customer_query: CustomerQuery, tool_message, tool_outputs: list, client: instructor.Instructor) -> SupportTicket:
    """Uses an instructor-patched client to generate the final SupportTicket."""
    print("\nStage 3: Generating final support ticket...")
    messages = [
        {"role": "user", "content": f"Here is the initial customer query: {customer_query.model_dump_json()}"},
        tool_message.model_dump() # Add the assistant's decision to call tools
    ]
    if tool_outputs:
        messages.extend(tool_outputs)

    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_model=SupportTicket,
    )
    print("Support ticket generation complete.")
    return final_response

# ==============================================================================
# 6. EXECUTION
# ==============================================================================
# Explanation:
# This block is the orchestrator. It runs the entire process sequentially,
# passing the output of each stage as the input to the next. It handles
# validating the raw input, executing the three AI stages, and running any
# tools the AI decides to use.
# ==============================================================================

# Initialize clients
openai_client = OpenAI()
instructor_client = instructor.from_openai(openai_client)

# --- Start Workflow ---
user_input_json = '''
{
    "name": "Joe User",
    "email": "joe@example.com",
    "query": "I'm really not happy with this product I bought",
    "order_id": "QWE-34567",
    "purchase_date": null
}
'''
validated_input = UserInput.model_validate_json(user_input_json)

# STAGE 1 EXECUTION
customer_query = classify_user_input(validated_input)

# STAGE 2 EXECUTION
tool_message = decide_on_tools(customer_query, openai_client)
tool_calls = tool_message.tool_calls
tool_outputs = []

if tool_calls:
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_tools[function_name]
        
        # Validate arguments before calling the tool
        if function_name == "lookup_faq_answer":
            args = FAQLookupArgs.model_validate_json(tool_call.function.arguments)
        elif function_name == "check_order_status":
            args = CheckOrderStatusArgs.model_validate_json(tool_call.function.arguments)
        else:
            continue

        output = function_to_call(args)
        tool_outputs.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": json.dumps(output),
        })

# STAGE 3 EXECUTION
support_ticket = generate_support_ticket(customer_query, tool_message, tool_outputs, instructor_client)

print("\n--- Final Structured Support Ticket ---")
print(support_ticket.model_dump_json(indent=2))