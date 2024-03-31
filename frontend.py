import tkinter as tk
from tkinter import filedialog
from groq_model_faiss import *
# TODO: define model options
model_options = ['mixtral-8x7b', 'llama-70b', 'gemma-7b']
groq_model = GroqLanguageModel()
# Functions setting variables...
def set_csv_path():
    csv_paths = filedialog.askopenfilenames(filetypes=[("Select CSV file", "*.csv")])
    for path in csv_paths:
        #selected_csv.set(path.split("/")[-1])
        selected_csv_path.set(path)
        selected_csv.set(path.split("/")[-1])
        groq_model.add_to_db(selected_csv_path.get())

def set_model(value):
    selected_model.set(value)

def submit_prompt():

    # Attempt at UI error handling
    if not selected_csv.get():
        error_message.set("Please select a CSV file")
        print("Please select a CSV file")
        return
    elif not selected_model.get():
        error_message.set("Please select a model")
        print("Please select a model")
        return
    elif prompt_entry.get() == "":
        error_message.set("Please enter a prompt")
        print("Please enter a prompt")
        return
    else:
        error_message.set("")
    
    
    # TODO: Run the prompt on both models and display the results in the text boxes
    # To retrieve info discussed in meeting, use widget getters...
    csv_path = selected_csv_path.get()
    model = selected_model.get()
    prompt = prompt_entry.get()

    # Set model to use:
    groq_model.set_model(model)
    # Get response and rag based response
    non_rag_response, rag_response = groq_model.get_dual_response(prompt)

    # update RAG and Non-RAG canvas
    rag_text.insert(tk.END, "Rag response after submitting response:\n" + "\n" + rag_response + "\n")
    nonrag_text.insert(tk.END, "Non-Rag response after submitting response:\n" + "\n" + non_rag_response + "\n")

# Create the root window...
root = tk.Tk()

root.title("Prompt Comparison Tool")
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=1)
root.columnconfigure(3, weight=1)
root.columnconfigure(4, weight=1)
root.columnconfigure(5, weight=1)
root.rowconfigure(0, weight=0)
root.rowconfigure(1, weight=0)
root.rowconfigure(2, weight=0)
root.rowconfigure(3, weight=2)

# Place widgets in the root window...
selected_csv = tk.StringVar(root)
selected_csv_path = tk.StringVar(root)

select_csv_button = tk.Button(root, text="Select CSV Files", command=set_csv_path)
select_csv_button.grid(row=0, column=0, padx=20, pady=10, sticky='ew') 
selected_csv_label = tk.Label(root, textvariable=selected_csv)
selected_csv_label.grid(row=1, column=0, padx=20, pady=10, sticky='w')

selected_model = tk.StringVar(root) 
select_model_dropdown = tk.OptionMenu(root, selected_model, *model_options, command=set_model)
select_model_dropdown.grid(row=0, column=1, padx=20, pady=10, sticky='ew')

prompt_entry = tk.Entry(root)
prompt_entry.grid(row=0, column=2, columnspan=3, padx=20, pady=5, sticky='ew')
prompt_entry.config(width=50)

error_message = tk.StringVar(root)
submit_button = tk.Button(root, text="Compare", command=submit_prompt)
submit_button.grid(row=0, column=5, padx=20, pady=10, sticky='ew')
error_message_label = tk.Label(root, textvariable=error_message)
error_message_label.grid(row=1, column=5, padx=20, pady=10, sticky='w')

rag_frame = tk.Frame(root, width=400, height=200, bg='white')
rag_frame.grid(row=3, column=0, columnspan=3, padx=20, pady=10, sticky='nsew')
rag_text = tk.Text(rag_frame, width=20, height=5, wrap=tk.WORD)
rag_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

nonrag_frame = tk.Frame(root, width=400, height=200, bg='white')
nonrag_frame.grid(row=3, column=3, columnspan=3, padx=20, pady=10, sticky='nsew')
nonrag_text = tk.Text(nonrag_frame, width=20, height=5, wrap=tk.WORD)
nonrag_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

root.mainloop()
