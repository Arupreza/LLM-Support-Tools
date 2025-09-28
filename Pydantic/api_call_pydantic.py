# ==============================================================================
# 1. IMPORTS
# ==============================================================================
# Explanation:
# We are importing the necessary components.
# - `BaseModel`, `Field`, `EmailStr`: Core parts of Pydantic for data validation and modeling.
#   `BaseModel` is the class you inherit from to create a model. `Field` lets you add extra
#   validation and metadata. `EmailStr` is a pre-built type for validating email addresses.
# - `List`, `Literal`, `Optional`: Standard Python typing hints to make our models precise.
# - `OpenAI`: The official client library to interact with the OpenAI API.
# - `instructor`: A library that patches the OpenAI client. Its purpose is to make the client
#   return Pydantic objects directly, ensuring the AI's output is structured correctly.
# - `load_dotenv`: A utility to load environment variables (like your API key) from a `.env` file.
#   This is a best practice for managing secret keys.
# - `date`: Standard Python type for handling dates.
# ==============================================================================
from pydantic import BaseModel, Field, EmailStr
from typing import List, Literal, Optional
from openai import OpenAI
import instructor
from dotenv import load_dotenv
from datetime import date


# ==============================================================================
# 2. Pydantic Model Definitions
# ==============================================================================
# Explanation:
# Models define the "shape" of your data. They ensure that any data created
# is valid and conforms to the types and constraints you specify.
#
# `UserInput`: Represents the raw input you expect from a user.
#   - `order_id`: This field is `Optional`, meaning it might not be present.
#     The `Field` function adds constraints: `ge=10000` (greater than or equal to)
#     and `le=99999` (less than or equal to) enforce a 5-digit number.
#
# `CustomerQuery`: Represents the structured data *after* AI analysis.
#   - It inherits from `UserInput`, so it automatically includes all fields from `UserInput`.
#   - `priority`, `category`, `tags`: These are the new fields the AI will generate.
#   - `Literal[...]`: Restricts the `category` field to one of three specific strings.
#     This forces the AI to classify the query correctly.
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
# 3. ENVIRONMENT AND DATA SETUP
# ==============================================================================
# Explanation:
# Before calling the AI, we prepare our data and environment.
# - `load_dotenv()`: Looks for a file named `.env` in your project directory
#   and loads variables like `OPENAI_API_KEY=...` into the environment.
# - `instructor.patch(OpenAI())`: This is the key step for `instructor`. It modifies
#   the standard `OpenAI` client, adding a `response_model` parameter to its
#   `chat.completions.create` method. This new parameter tells the client to
#   return a Pydantic object instead of a standard dictionary.
# - `user_input_json`: This is our raw, unstructured data, simulating a request
#   from a web form or API. Notice the field is "order_number", which doesn't
#   match our model's "order_id". Pydantic will handle this if configured, but here
#   it would fail. I have corrected it to `order_id` to match the model.
# ==============================================================================
load_dotenv()

# Patch the OpenAI client to add the `response_model` parameter
client = instructor.patch(OpenAI())

# Raw input data, simulating what you might get from an external source.
# Note: The key "order_id" must match the Pydantic model field name.
user_input_json = '''{
    "name": "Joe User",
    "email": "joe.user@example.com",
    "query": "I ordered a new computer monitor and it arrived with the screen cracked. This is the second time this has happened. I need a replacement ASAP.",
    "order_id": 12345,
    "purchase_date": "2025-09-27"
}'''

# Validate the raw input against our base model. This is a good practice to
# catch errors early, before sending data to the AI.
user_input = UserInput.model_validate_json(user_input_json)


# ==============================================================================
# 4. AI INFERENCE
# ==============================================================================
# Explanation:
# This is where we call the AI model.
# - `prompt`: We create a clear, instructional prompt for the AI. Including the
#   validated `user_input` object ensures the AI has all necessary context.
# - `client.chat.completions.create()`: The standard method to call the model.
# - `response_model=CustomerQuery`: This is the crucial part added by `instructor`.
#   You are telling the AI: "Your final response MUST conform to the structure of
#   the `CustomerQuery` Pydantic model." Instructor handles the logic to force
#   the model into generating valid JSON that fits this schema.
#
# The `response` variable will not be a raw API response. It will be an
# instance of your `CustomerQuery` class, fully parsed and validated.
# ==============================================================================
prompt = (
    f"Analyze the following customer query and extract the relevant information. "
    f"Classify its priority, category, and generate relevant tags. "
    f"Query details: {user_input.model_dump_json(indent=2)}"
)

# Call the AI and get a structured, validated Pydantic object back.
# The `instructor` library guarantees the output will match the `CustomerQuery` model.
response: CustomerQuery = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    response_model=CustomerQuery,
)


# ==============================================================================
# 5. OUTPUT AND VERIFICATION
# ==============================================================================
# Explanation:
# The `response` object is already a validated Pydantic model, so you can
# directly access its fields like `response.priority` or `response.category`.
# Using `model_dump_json` provides a clean, readable JSON output for verification.
# ==============================================================================
print("AI Structured Response:")
print(response.model_dump_json(indent=2))

# You can now access the structured data with confidence.
print("\n--- Accessing individual fields ---")
print(f"Priority: {response.priority}")
print(f"Category: {response.category}")
print(f"Tags: {response.tags}")


# ==============================================================================
# 6. Alternative Method: `pydantic-ai`
# ==============================================================================
# Explanation:
# Your original code also included `pydantic-ai`. This is another library that
# achieves a similar goal but with a different, higher-level abstraction.
# It uses an "Agent" concept.
#
# from pydantic_ai import Agent
#
# agent = Agent(
#     model="openai:gpt-4o",
#     output_type=CustomerQuery
# )
#
# response_from_agent = agent.run_sync(prompt)
#
# This can be simpler for some use cases but gives you less direct control over
# the API call parameters compared to using `instructor` with the native client.
# Both are valid tools for achieving structured AI output.
# ==============================================================================