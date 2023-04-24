import openai

openai.api_key = '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'

model_engine = 'text-curie-001'

prompt = 'What is the girl hair color using the following description: A girl was riding a red bike in the woods ' \
         'while she ' \
         'was eating a green apple some one saw her lovely yellow hair'

params = {
    'engine': model_engine,
    'prompt': prompt,
    'temperature': 0,
    'max_tokens': 1000,
    'top_p': 1,
    'frequency_penalty': 0,
    'presence_penalty': 0,
}

response = openai.Completion.create(**params)
answer = response.choices[0].text.strip()

print(answer)
