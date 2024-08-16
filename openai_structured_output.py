import gradio as gr
import pandas as pd
import os
import json
import time
from langchain_unstructured import UnstructuredLoader
from openai import OpenAI

response_history = pd.DataFrame(columns=["file_name", "Likelihood", "Decision"])
previous_file = None

def chatbot_openai(api_key, system, user, filepath):
    os.environ["OPENAI_API_KEY"] = api_key

    file_paths = [
    filepath,
    ]

    loader = UnstructuredLoader(file_paths)
    data = loader.load()
    
    d={"Likelihood":"", "Decision":""}

    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": f"{system} \n Answer the question of the user on the basis of provided context in the Structured JSON format as provided below:\n {d}."},
        {"role": "user", "content": f"{user} \n Answer the question on the basis of following context:\n\n{data}"},
    ]
    )
    output = response.choices[0].message.content
    final = json.loads(output)

    return final

def chatbot(api_key, system_prompt, user_prompt, num_responses, file=None):
    global response_history, previous_file

    if file is not None:
        file_path = file.name

        if file_path != previous_file:

            if os.path.exists("response.xlsx"):
                os.remove("response.xlsx")
            
            response_history = pd.DataFrame(columns=["file_name", "Likelihood", "Decision"])
            previous_file = file_path 
            
    else:
        file_path = None

    for _ in range(num_responses):
        response = chatbot_openai(api_key, system_prompt, user_prompt, file_path)
        likelihood = response['Likelihood']
        decision = response['Decision']
        new_entry = pd.DataFrame({
            "file_name": [file_path],
            "Likelihood": [likelihood],
            "Decision": [decision],
        })
        response_history = pd.concat([response_history, new_entry], ignore_index=True)
        time.sleep(3)

    return response

def download_excel():
    global response_history
    response_history.to_excel("response.xlsx", index=False)
    return "response.xlsx"


default_system_prompt = "You are a juror in a legal case. Please, read the evidence in this case and respond on a scale from 0-100 how likely you think the defendant is guilty. Also provide a dichotomous guilty/innocent decision regarding the defendant. The decision should be either 'Innocent' or 'Guilty'."
default_user_prompt = "Please, read the evidence in this case and respond on a scale from 0-100 how likely you think the defendant is guilty. Also provide a dichotomous decision either the denfendent is 'Innocent' or 'Guilty'."

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
        system_prompt_input = gr.Textbox(label="System Prompt", value=default_system_prompt)
        user_prompt_input = gr.Textbox(label="User Prompt", value=default_user_prompt)
        num_responses_input = gr.Number(label="Number of Iterations", value=1, precision=0, maximum=500)  # Add limit with maximum=500

    with gr.Row():
        file_input = gr.File(label="Upload File")

    response_output = gr.Textbox(label="Response")

    with gr.Row():
        submit_button = gr.Button("Submit")
        download_button = gr.Button("Download Responses in Excel file")

    submit_button.click(chatbot, [api_key_input, system_prompt_input, user_prompt_input, num_responses_input, file_input], response_output)
    download_button.click(download_excel, outputs=gr.File())

demo.launch(debug=True)
