import requests
import json
from langchain_unstructured import UnstructuredLoader
from openai import OpenAI
import gradio as gr
import os
import pandas as pd
import time


response_history = pd.DataFrame(columns=["file_name", "GPT-Response"])
previous_file = None

# Function to structure output using OpenAI's API
def structured_output(openai_api, response):
    client = OpenAI(api_key=openai_api)
    d = {"Likelihood": "", "Decision": ""}

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": f"Answer the question of the user on the basis of provided context in the Structured JSON format as provided below:\n {d}."},
            {"role": "user", "content": f"Extract the Likelihood (out of 100), and defendant's Decision either he/she is guilty or innocent from the provided context. Context:\n\n{response}"},
        ]
    )
    output = response.choices[0].message.content
    final = json.loads(output)
    return final

# Function to handle chatbot interaction
def claude_chatbot(openai_api, api_key, model, system, user, filepath):

    file_paths = [
        filepath,
    ]
    loader = UnstructuredLoader(file_paths)
    data = loader.load()
    print(len(data))
    if model == 'Llama':
        model_name = 'meta-llama/llama-3.1-405b-instruct'
    if model == 'Claude':
        model_name = 'anthropic/claude-3.5-sonnet'
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        data=json.dumps({
            "model": model_name,
            "messages": [
                {"role": "system", "content": f"{system} \n Answer the question of the user on the basis of provided context."},
                {"role": "user", "content": f"{user} \n Answer the question on the basis of following context:\n\n{data}"},
            ],
            "top_p": 1,
            "temperature": 0.8,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1.1,
            "top_k": 0,
        })
    )
    output = json.loads(response.text)
    final = output['choices'][0]['message']['content']
    print("response", final)

    structured = structured_output(openai_api, final)
    return structured, final

# Main chatbot function to handle responses
def chatbot(openai_api, api_key, model, system_prompt, user_prompt, num_responses, file=None):
    global response_history, previous_file

    if file is not None:
        file_path = file.name

        if file_path != previous_file:
            if os.path.exists("response.xlsx"):
                os.remove("response.xlsx")

            response_history = pd.DataFrame(columns=["file_name", "Likelihood", "Decision", "Response", "Model"])
            previous_file = file_path

    else:
        file_path = None

    for _ in range(num_responses):
        response, complete_response = claude_chatbot(openai_api, api_key, model, system_prompt, user_prompt, file_path)
        likelihood = response['Likelihood']
        decision = response['Decision']
        new_entry = pd.DataFrame({
            "file_name": [file_path],
            "Likelihood": [likelihood],
            "Decision": [decision],
            "Response": [complete_response],
            "Model": [model]
        })
        response_history = pd.concat([response_history, new_entry], ignore_index=True)
        time.sleep(3)

    return complete_response

# Function to download responses as an Excel file
def download_excel():
    global response_history
    response_history.to_excel("response.xlsx", index=False)
    return "response.xlsx"

# Default prompts
default_system_prompt = "You are a juror in a legal case. Please, read the evidence in this case and respond on a scale from 0-100 how likely you think the defendant is guilty. Also provide a dichotomous guilty/innocent decision regarding the defendant. The decision should be either 'Innocent' or 'Guilty'."
default_user_prompt = "Please, read the evidence in this case and respond on a scale from 0-100 how likely you think the defendant is guilty. Also provide a dichotomous decision either the defendant is 'Innocent' or 'Guilty'."

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
        api_key_input1 = gr.Textbox(label="Openrouter API Key", type="password")
        model = gr.Dropdown(choices=["Llama", "Claude"], label="Select an Option")
        system_prompt_input = gr.Textbox(label="System Prompt", value=default_system_prompt)
        user_prompt_input = gr.Textbox(label="User Prompt", value=default_user_prompt)
        num_responses_input = gr.Number(label="Number of Iterations", value=1, precision=0, maximum=500)  # Add limit with maximum=500

    with gr.Row():
        file_input = gr.File(label="Upload File")

    response_output = gr.Textbox(label="Response")

    with gr.Row():
        submit_button = gr.Button("Submit")
        download_button = gr.Button("Download Responses in Excel file")

    submit_button.click(chatbot, [api_key_input, api_key_input1, model, system_prompt_input, user_prompt_input, num_responses_input, file_input], response_output)
    download_button.click(download_excel, outputs=gr.File())

demo.launch(debug=True)
