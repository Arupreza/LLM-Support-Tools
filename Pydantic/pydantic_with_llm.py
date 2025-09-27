# ==============================================================================
# 1. IMPORTS AND SETUP
# ==============================================================================
# Import necessary libraries.
from pydantic import BaseModel, ValidationError, Field, EmailStr
from typing import List, Literal, Optional
from datetime import date
from dotenv import load_dotenv
import openai
import json

# Load environment variables from a .env file (for OPENAI_API_KEY).
load_dotenv()

# Initialize the OpenAI client. It automatically finds the API key from the
# environment variables you loaded.
client = openai.OpenAI()


# ==============================================================================
# 2. DEFINE DATA MODELS (THE "SCHEMA")
# ==============================================================================
# We define two Pydantic models. These classes serve as the schema for our
# expected data structure. Pydantic uses these models to validate incoming data.

class UserInput(BaseModel):
    """A base model for the initial user query."""
    name: str
    email: EmailStr
    query: str
    order_id: Optional[int] = Field(
        None,
        description="5-digit order number",
        ge=10,
        le=99999
    )
    purchase_date: Optional[date] = None


class CustomerQuery(UserInput):
    """
    An enhanced model that inherits from UserInput and adds fields for the
    LLM to populate. This is the final, structured data we want.
    """
    priority: Literal['low', 'medium', 'high'] = Field(
        ..., description="Priority level determined by the LLM."
    )
    category: Literal['refund_request', 'information_request', 'other'] = Field(
        ..., description="The category of the user's query."
    )
    is_complaint: bool = Field(
        ..., description="Whether the query is a complaint."
    )
    tags: List[str] = Field(
        ..., description="A list of relevant keyword tags for the query."
    )


# ==============================================================================
# 3. DEFINE CORE FUNCTIONS
# ==============================================================================

def call_llm(prompt: str, model: str = "gpt-4o"):
    """
    Sends a prompt to the OpenAI API and returns the content of the response.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # Ensure the model returns a JSON object.
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        return None


def validate_with_model(data_model: BaseModel, llm_response: str):
    """
    Tries to validate the LLM's JSON response against a Pydantic model.
    """
    try:
        # `model_validate_json` is a powerful Pydantic method that parses a
        # JSON string and validates it against the model in one step.
        validated_data = data_model.model_validate_json(llm_response)
        print("✅ LLM response validation successful!")
        return validated_data, None
    except ValidationError as e:
        # If validation fails, Pydantic provides a detailed error message.
        print(f"❌ Validation error. The LLM response did not match the schema.")
        return None, str(e)


def create_retry_prompt(original_prompt: str, original_response: str, error_message: str) -> str:
    """
    Creates a new prompt asking the LLM to fix its previous, invalid response.
    """
    retry_prompt = f"""
The previous attempt to generate a JSON object failed due to a validation error.
Your task is to correct the JSON object based on the error message provided.

Here was the original request:
<original_prompt>
{original_prompt}
</original_prompt>

Here was the invalid JSON response you provided:
<invalid_response>
{original_response}
</invalid_response>

And here is the validation error that occurred:
<error_message>
{error_message}
</error_message>

Please correct the JSON object and return ONLY the valid JSON, without any additional text or explanations.
"""
    return retry_prompt


# ==============================================================================
# 4. MAIN VALIDATION AND RETRY LOGIC
# ==============================================================================

def validate_llm_response_with_retries(prompt: str, data_model: BaseModel, n_retry: int = 5, model: str = "gpt-4o"):
    """
    The main function that calls the LLM, validates the response, and retries
    on failure.
    """
    current_prompt = prompt
    last_error = ""

    for attempt in range(n_retry + 1):
        print(f"\n--- Attempt {attempt + 1} of {n_retry + 1} ---")
        if attempt > 0:
            print("Retrying with new prompt to fix previous error...")

        response_content = call_llm(current_prompt, model=model)
        if not response_content:
            last_error = "API call failed."
            continue # Try again if the API call fails

        validated_data, validation_error = validate_with_model(data_model, response_content)

        if validated_data:
            return validated_data, None  # Success!

        if validation_error:
            last_error = validation_error
            # If validation fails, create a new prompt that asks the LLM to fix the error.
            current_prompt = create_retry_prompt(
                original_prompt=prompt,
                original_response=response_content,
                error_message=validation_error
            )
    
    # If all retries fail
    print(f"\n❌ Maximum retries reached. Unable to get a valid response.")
    return None, last_error


# ==============================================================================
# 5. EXECUTION
# ==============================================================================

# --- Step 1: Define the initial user input ---
# This is the raw data we start with, perhaps from a web form or API.
user_input_data = {
    "name": "Joe User",
    "email": "joe.user@example.com",
    "query": "I bought a laptop carrying case and it turned out to be the wrong size. It arrived yesterday. The order number is 12345. I need to return it.",
    "order_id": 12345,
    "purchase_date": date(2025, 9, 27) # Using a relevant date
}
# Create an instance of our base model.
user_input_model = UserInput(**user_input_data)


# --- Step 2: Create the initial prompt using Pydantic's schema generation ---
# `model_json_schema()` generates a detailed JSON Schema definition from your
# Pydantic model. This is far more precise than providing a simple example.
data_model_schema = json.dumps(CustomerQuery.model_json_schema(), indent=2)

initial_prompt = f"""
Please analyze the following user query and extract the required information.
Return your analysis as a JSON object that strictly follows this schema:
<json_schema>
{data_model_schema}
</json_schema>

Here is the user's query data:
<user_query>
{user_input_model.model_dump_json(indent=2)}
</user_query>

Respond ONLY with the valid JSON object. Do not include any explanations,
markdown formatting, or other text outside of the JSON structure.
"""

# --- Step 3: Run the validation process ---
print("🚀 Starting LLM call and validation process...")
final_validated_data, final_error = validate_llm_response_with_retries(
    prompt=initial_prompt,
    data_model=CustomerQuery,
    n_retry=2 # Try the initial call, plus 2 retries
)

# --- Step 4: Display the final result ---
if final_validated_data:
    print("\n\n✅✅✅ Successfully validated LLM response after processing! ✅✅✅")
    print(final_validated_data.model_dump_json(indent=2))
else:
    print(f"\n\n❌❌❌ Process failed after all retries. ❌❌❌")
    print(f"Last known error: {final_error}")