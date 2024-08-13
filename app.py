import gradio as gr
import pandas as pd
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import UnstructuredWordDocumentLoader

# Initialize response history DataFrame
response_history = pd.DataFrame(columns=["file_name", "GPT-Response"])

# Function to interact with OpenAI GPT-4
def chatbot_openai(api_key, system, user, filepath):
    os.environ["OPENAI_API_KEY"] = api_key  # Set the API key as an environment variable
    
    chat = ChatOpenAI(model="gpt-4", temperature=0.2)
    history = ChatMessageHistory()

    loader = UnstructuredWordDocumentLoader(filepath)
    data = loader.load()

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """{system}
                  Answer the user's questions based on the below context:\n\n{context}""",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(chat, question_answering_prompt)
    history.add_user_message(f"""On the basis of the context provided, answer the following:

    ```
    {user}
    ```

    """)

    response = document_chain.invoke(
        {
            "messages": history.messages,
            "context": data,
            "system": system,
            "user": user,
        }
    )
    return response

# Function to handle Gradio interface interaction
def chatbot(api_key, system_prompt, user_prompt, file=None):
    global response_history
    
    if file is not None:
        file_path = file.name
    else:
        file_path = None

    response = chatbot_openai(api_key, system_prompt, user_prompt, file_path)

    new_entry = pd.DataFrame({
        "file_name": [file_path],
        "GPT-Response": [response]
    })
    response_history = pd.concat([response_history, new_entry], ignore_index=True)
    
    return response

# Function to download chat history as an Excel file
def download_excel():
    global response_history
    response_history.to_excel("response.xlsx", index=False)
    return "response.xlsx"

# Default values for system and user prompts
default_system_prompt = "You are a juror in a legal case. Please, read the evidence in this case and respond on a scale from 0-100 how likely you think the defendant is guilty. Also provide a dichotomous guilty/innocent decision regarding the defendant."
default_user_prompt = "Please, read the evidence in this case and respond on a scale from 0-100 how likely you think the defendant is guilty. Also provide a dichotomous guilty/innocent decision regarding the defendant."

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
        system_prompt_input = gr.Textbox(label="System Prompt", value=default_system_prompt)
        user_prompt_input = gr.Textbox(label="User Prompt", value=default_user_prompt)
    
    with gr.Row():
        file_input = gr.File(label="Upload File")
    
    response_output = gr.Textbox(label="Response")
    
    with gr.Row():
        submit_button = gr.Button("Submit")
        download_button = gr.Button("Download Responses in Excel file")
    
    submit_button.click(chatbot, [api_key_input, system_prompt_input, user_prompt_input, file_input], response_output)
    download_button.click(download_excel, outputs=gr.File())

# Launch the Gradio app
demo.launch(debug=True)
