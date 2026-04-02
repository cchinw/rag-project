import os
from dotenv import load_dotenv
import anthropic


#Load env variables from .env
load_dotenv()

#Init Anthropic Client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

#send a smple test message
message = client.messages.create(
  model="claude-opus-4-5",
  max_tokens=1024,
  messages=[
    {"role": "user", "content": "Say hello and confirm you are working."}
  ]
)

print(message.content[0].text)