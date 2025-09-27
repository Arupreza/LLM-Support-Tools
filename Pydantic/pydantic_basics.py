from pydantic import BaseModel, ValidationError, EmailStr
import json

class UserInput(BaseModel):
    name: str
    email: EmailStr
    query: str


user_input_0 = UserInput(
    name = 'Reza User',
    email = 'arupreza@sch.ac.kr',
    query = 'I forgot my password'
)

print(user_input_0)
print("\n")

# user_input_1 = UserInput(
#     name = 'Reza User',
#     email = 'no email',
#     query = 'I forgot my password'
# )

# print(user_input_1)

def validation_user_input(input_data):
    try:
        user_input = UserInput(**input_data)
        print(f"✅ Valid user input created:")
        print(f"{user_input.model_dump_json(indent=2)}")
        return user_input
    except ValidationError as e:
        print(f"❌ Validation error occurred:")
        for error in e.errors():
            print(f"- {errors['loc'][0]}: {errors['msg']}")
        return None

user_input_acc = {
    "name": "Reza User",
    "email": "arupreza@sch.ac.kr",
    "query": "I forgot my password"
}

user_input = validation_user_input(user_input_acc)

print(user_input)
print("\n")


user_input_prob = {
    "name": "Reza User",
    "email": "arupreza@sch.ac.kr"
}

user_input = validation_user_input(user_input_prob)

print(user_input)
print("\n")