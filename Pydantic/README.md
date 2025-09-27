# Robust LLM Data Extraction with Pydantic Validation and Auto-Retry

## Overview

Extract structured data from unstructured text using LLMs with guaranteed schema compliance. This implementation combines Pydantic's validation with automatic error correction, creating a resilient pipeline that handles common LLM output inconsistencies.

## Key Features

- **Schema-driven extraction**: Pydantic models define exact output structure and constraints
- **Auto-generated prompts**: JSON Schema generation eliminates manual prompt crafting
- **Validation-first approach**: Immediate output validation with detailed error reporting  
- **Self-correcting retries**: Failed validations trigger contextual correction prompts
- **Type safety**: Full static typing with runtime validation

## Architecture

The system implements a validation-retry loop:

1. **Schema Definition**: Pydantic models specify expected data structure
2. **Prompt Generation**: Automatic JSON Schema creation for LLM instructions
3. **LLM Query**: Structured request with schema constraints
4. **Validation**: Immediate Pydantic model validation
5. **Error Correction**: Contextual retry prompts on validation failure

## Installation

```bash
# Clone repository
git clone <repository-url>
cd robust-llm-extraction

# Install dependencies
pip install pydantic python-dotenv openai

# Configure API key
echo "OPENAI_API_KEY=sk-your-api-key" > .env
```

## Usage

```python
from pydantic_with_llm import validate_llm_response_with_retries

# Define your schema
class CustomerQuery(UserInput):
    intent: Literal["question", "complaint", "request"]
    priority: int = Field(ge=1, le=5)
    entities: List[str]

# Extract structured data
result = validate_llm_response_with_retries(
    user_input="I'm having trouble with my order #12345",
    model_class=CustomerQuery
)
```

## Implementation Details

### Core Components

- **UserInput/CustomerQuery**: Pydantic models serving as schema definitions
- **validate_llm_response_with_retries()**: Main orchestration with retry logic
- **create_retry_prompt()**: Error-aware prompt generation for corrections

### Error Handling

The system handles common LLM failure modes:
- Invalid JSON syntax
- Missing required fields  
- Type mismatches
- Constraint violations

Each failure triggers a targeted retry with specific error context, improving correction success rates.

### Performance Considerations

- Schema validation is O(1) for most field types
- Retry attempts are bounded (default: 3 attempts)
- Network calls are minimized through efficient error messaging

## Limitations

- Dependent on LLM reasoning capabilities for error correction
- Token usage increases with retry attempts
- Complex nested schemas may require prompt engineering
- No streaming support for large responses

## Contributing

Contributions welcome. Focus areas:
- Additional LLM provider support
- Streaming response handling
- Advanced retry strategies
- Performance optimizations