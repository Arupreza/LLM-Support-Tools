# ==============================================================================
# 1. IMPORTS
# ==============================================================================
# Explanation:
# These are your tools.
# - `pydantic`: For creating strict data schemas. This is how you define the
#   exact structure you want your data to have. `BaseModel` is the foundation,
#   `Field` adds rules, and `EmailStr` is a pre-built validator.
# - `typing`: Standard Python tools to define data shapes precisely. `List`,
#   `Literal`, and `Optional` make your Pydantic models unambiguous.
# - `openai`: The client library for communicating with any server that uses
#   an OpenAI-compatible API. You configure it to point to a target URL.
# - `instructor`: A library that patches the `openai` client. Its only job is to
#   force the AI's JSON output to perfectly match your Pydantic model, retrying
#   on failure. It guarantees structure.
# - `datetime.date`: The standard Python object for handling dates.
# ==============================================================================
from pydantic import BaseModel, Field, EmailStr
from typing import List, Literal, Optional
from openai import OpenAI
import instructor
from datetime import date


# ==============================================================================
# 2. Pydantic Model Definitions
# ==============================================================================
# Explanation:
# This is your contract. You are defining the required structure for both the
# initial input and the final, AI-enriched output.
#
# `UserInput`: Defines the shape of the raw data you receive. It ensures the
# input is valid before you waste a call to the AI.
#
# `CustomerQuery`: This is the critical model. It inherits all fields from
# `UserInput` and adds the new fields you expect the AI to generate. This
# schema is given to `instructor` to enforce upon the model's response.
# Using `Literal` is a powerful way to force the AI to classify the query
# into one of your predefined categories.
# ==============================================================================
class UserInput(BaseModel):
    name: str
    email: EmailStr
    query: str
    order_id: Optional[int] = Field(
        None,
        description="5-digit order number (cannot start with 0)",
        ge=10000,
        le=99999,
    )
    purchase_date: Optional[date] = None


class CustomerQuery(UserInput):
    priority: str = Field(
        ..., description="Priority level: low, medium, or high"
    )
    category: Literal[
        'refund_request', 'information_request', 'other'
    ] = Field(..., description="The main category of the customer's query.")
    tags: List[str] = Field(..., description="A list of relevant keyword tags for this query.")


# ==============================================================================
# 3. CLIENT CONFIGURATION AND DATA PREPARATION
# ==============================================================================
# Explanation:
# Here, you prepare the connection to the server and validate your input data.
# - `OpenAI(...)`: You instantiate the client.
# - `base_url="http://localhost:11434/v1"`: This is the critical line. You are
#   directing the client to send its requests to the Ollama server running on
#   your local machine, not to the official OpenAI servers.
# - `api_key="ollama"`: This is a placeholder. The local Ollama server does not
#   require authentication, but the client library requires a value.
# - `instructor.patch()`: This modifies the client instance, adding the
#   `response_model` capability.
# - `UserInput.model_validate_json()`: You parse and validate the raw input
#   string against your `UserInput` schema. This ensures data integrity
#   before proceeding.
# ==============================================================================
client = instructor.patch(
    OpenAI(
        base_url="http://localhost:11434/v1",  # Points to the local Ollama server
        api_key="ollama"                       # Placeholder for Ollama
    )
)

user_input_json = '''{
    "name": "Joe User",
    "email": "joe.user@example.com",
    "query": "I ordered a new computer monitor and it arrived with the screen cracked. This is the second time this has happened. I need a replacement ASAP.",
    "order_id": 12345,
    "purchase_date": "2025-09-27"
}'''

user_input = UserInput.model_validate_json(user_input_json)


# ==============================================================================
# 4. AI INFERENCE
# ==============================================================================
# Explanation:
# This is the actual call to the model via the server.
# - `prompt`: A clear, direct instruction to the AI, including the validated
#   input data formatted as a JSON string for clarity.
# - `client.chat.completions.create()`: The function that sends the request.
# - `model="llama3.1:8b"`: You are requesting the model that the Ollama server
#   has loaded under this specific name. This is not a file path. If the server
#   is not running this model, this call will fail.
# - `response_model=CustomerQuery`: This is the `instructor` magic. You are
#   commanding the client to return a valid `CustomerQuery` Pydantic object.
#   Instructor will handle the communication with the model to ensure its
#   output conforms to this structure. The `response` variable will be a
#   `CustomerQuery` object, not a dictionary or raw text.
# ==============================================================================
prompt = (
    f"Analyze the following customer query and extract the relevant information. "
    f"Classify its priority, category, and generate relevant tags. "
    f"Query details: {user_input.model_dump_json(indent=2)}"
)

response: CustomerQuery = client.chat.completions.create(
    model="llama3.1:8b",  # The model name as registered and served by Ollama
    messages=[{"role": "user", "content": prompt}],
    response_model=CustomerQuery,
)


# ==============================================================================
# 5. OUTPUT AND VERIFICATION
# ==============================================================================
# Explanation:
# The operation is complete. The `response` variable is a fully validated and
# structured Pydantic object. You can access its fields with dot notation
# (e.g., `response.priority`) with the guarantee that they exist and have the
# correct data type.
# - `model_dump_json(indent=2)`: A Pydantic method to cleanly print the entire
#   structured object as a formatted JSON string for verification.
# ==============================================================================
print("AI Structured Response from Local Llama 3.1:")
print(response.model_dump_json(indent=2))

print("\n--- Accessing individual fields ---")
print(f"Priority: {response.priority}")
print(f"Category: {response.category}")
print(f"Tags: {response.tags}")