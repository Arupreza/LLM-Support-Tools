# ==============================================================================
# 1. IMPORTS
# ==============================================================================
# Import necessary components from the Pydantic library and standard libraries.
from pydantic import (
    BaseModel,          # The core class your data models will inherit from.
    ValidationError,    # The exception raised when validation fails.
    EmailStr,           # A special string type that validates email formats.
    Field               # A function to add extra validation and metadata to fields.
)
from typing import Optional # Used for fields that are not required.
from datetime import date   # A standard type for holding date values.

# ==============================================================================
# 2. DATA MODEL DEFINITION
# ==============================================================================
# Define a Pydantic model by creating a class that inherits from BaseModel.
# This class acts as a schema for your data. It defines the expected fields,
# their data types, and any validation rules.

class UserInput(BaseModel):
    """
    A model to validate user support queries. It defines the structure
    and rules for incoming data.
    """
    # --- Required Fields ---
    # These fields must be present in the input data.
    name: str
    email: EmailStr  # Pydantic will automatically validate this is a valid email.
    query: str

    # --- Optional Fields with Advanced Validation ---
    # `Optional[type]` means the field is not required.
    # `Field()` is used to add more constraints and documentation.
    order_id: Optional[int] = Field(
        default=None,  # The default value if not provided.
        description="5-digit order number (cannot start with 0)",
        ge=10,      # ge = "greater than or equal to". Ensures a 5-digit number.
        le=99999       # le = "less than or equal to". Ensures a 5-digit number.
    )
    purchase_date: Optional[date] = None # A simple optional field with a default of None.

# ==============================================================================
# 3. VALIDATION FUNCTION
# ==============================================================================
# This function encapsulates the validation logic. It takes raw data (as a
# dictionary) and attempts to create an instance of your Pydantic model.

def validate_user_input(input_data: dict):
    """
    Validates a dictionary against the UserInput model.

    Args:
        input_data: A dictionary containing the data to validate.

    Returns:
        An instance of UserInput if validation is successful, otherwise None.
    """
    try:
        # The core of Pydantic validation. The `**input_data` syntax unpacks the
        # dictionary into keyword arguments for the UserInput constructor.
        # If the data matches the model's schema, an instance is created.
        # If not, a ValidationError is raised.
        user_input_instance = UserInput(**input_data)

        print("✅ Validation successful. Model created:")
        # .model_dump_json() serializes the validated data into a JSON string.
        # `indent=2` makes the JSON output human-readable.
        print(user_input_instance.model_dump_json(indent=2))
        return user_input_instance

    except ValidationError as e:
        # This block executes only if the input data fails validation.
        # The exception object `e` contains detailed information about the errors.
        print(f"❌ Validation failed with {len(e.errors())} error(s):")

        # `e.errors()` returns a list of dictionaries, each describing one error.
        for error in e.errors():
            # `error['loc']` is a tuple of the field name (e.g., ('email',)).
            # `error['msg']` is a human-readable description of the error.
            field_name = error['loc'][0]
            error_message = error['msg']
            print(f"- Field '{field_name}': {error_message}")
        return None

# ==============================================================================
# 4. USAGE EXAMPLES
# ==============================================================================
# Here, we test the validation function with different sets of data to see how
# the model behaves with valid and invalid inputs.

print("--- SCENARIO 1: Valid data with all fields ---")
valid_data_full = {
    "name": "Reza User",
    "email": "arupreza@sch.ac.kr",
    "query": "I need to return an item.",
    "order_id": 54321,
    "purchase_date": date(2025, 9, 27)
}
validate_user_input(valid_data_full)
print("-" * 50)


print("\n--- SCENARIO 2: Valid data with only required fields ---")
valid_data_required = {
    "name": "Jane Doe",
    "email": "jane.doe@example.com",
    "query": "I forgot my password."
}
validate_user_input(valid_data_required)
print("-" * 50)


print("\n--- SCENARIO 3: Invalid data - missing a required field ('query') ---")
invalid_data_missing_field = {
    "name": "John Smith",
    "email": "john.smith@example.com"
}
validate_user_input(invalid_data_missing_field)
print("-" * 50)


print("\n--- SCENARIO 4: Invalid data - bad email format and wrong type ---")
invalid_data_bad_format = {
    "name": "Test User",
    "email": "not-an-email",  # This will fail EmailStr validation.
    "query": "This is a test.",
    "order_id": "not-a-number" # This will fail type validation (str vs int).
}
validate_user_input(invalid_data_bad_format)
print("-" * 50)


print("\n--- SCENARIO 5: Invalid data - order_id out of range ---")
invalid_data_out_of_range = {
    "name": "Another User",
    "email": "another.user@example.com",
    "query": "My order ID is wrong.",
    "order_id": 123  # This will fail the `ge=10000` constraint.
}
validate_user_input(invalid_data_out_of_range)
print("-" * 50)