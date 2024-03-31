import os
from openai import OpenAI

client = OpenAI(api_key="")
import random
import pandas as pd

def generate_example(prompt, prev_examples, temperature=.5):
    messages=[
        {
            "role": "system",
            "content": f"You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{prompt}`"
        }
    ]

    if len(prev_examples) > 0:
        if len(prev_examples) > 10:
            prev_examples = random.sample(prev_examples, 10)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = client.chat.completions.create(model="gpt-4",
    messages=messages,
    temperature=temperature,
    max_tokens=1354)

    return response.choices[0].message.content

def generate_system_message(prompt, temperature=.5):
    response = client.chat.completions.create(model="gpt-4",
    messages=[
      {
        "role": "system",
        "content": "You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. A good format to follow is `Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO.`.\n\nMake it as concise as possible. Include nothing but the system prompt in your response.\n\nFor example, never write: `\"$SYSTEM_PROMPT_HERE\"`.\n\nIt should be like: `$SYSTEM_PROMPT_HERE`."
      },
      {
          "role": "user",
          "content": prompt.strip(),
      }
    ],
    temperature=temperature,
    max_tokens=500)

    return response.choices[0].message.content

def generate_data(prompt, temperature, number_of_examples):

    prev_examples = []
    for i in range(number_of_examples):
        print(f'Generating example {i}')
        example = generate_example(prompt, prev_examples, temperature)
        prev_examples.append(example)

    system_message = generate_system_message(prompt)

    prompts = []
    responses = []

    for example in prev_examples:
        try:
            split_example = example.split('-----------')
            prompts.append(split_example[1].strip())
            responses.append(split_example[3].strip())
        except:
            pass

    df = pd.DataFrame({
        'prompt': prompts,
        'response': responses
    })

    df = df.drop_duplicates()

    train_df = df.sample(frac=0.9, random_state=42)
    test_df = df.drop(train_df.index)

    train_df.to_json('content/train.jsonl', orient='records', lines=True)
    test_df.to_json('content/test.jsonl', orient='records', lines=True)

    return system_message