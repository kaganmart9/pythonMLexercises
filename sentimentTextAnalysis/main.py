import tkinter as tk
from tkinter import messagebox
from textblob import TextBlob

#analyzing the text and categorizing the text
def analyze_text():
    user_text = text_entry.get()
    blob = TextBlob(user_text)
    sentiment = blob.sentiment.polarity
    if -1 < sentiment < -0.6:
        result = "It is a highly negative text."
    elif -0.6 <= sentiment < -0.2:
        result = "It is a negative text."
    elif -0.2 <= sentiment < 0.2:
        result = "It is a neutral text."
    elif 0.2 <= sentiment < 0.6:
        result = "It is a positive text."
    else:
        result = "It is a highly positive text."

    messagebox.showinfo("Sentiment Analysis Result", result) #showing result in a window pop-up

def clear_text():
    text_entry.delete(0, tk.END)

# Create the main window
root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("400x300") #setting the window size

# Add a title label
title_label = tk.Label(root, text="Enter the Text", font=("Helvetica", 16))
title_label.pack(pady=20)  #padding for label

# Add a text input box (centered)
text_entry = tk.Entry(root, font=("Helvetica", 12))
text_entry.pack(pady=10, padx=20, ipadx=20, ipady=10)  #padding and size adjusting

#Adding buttons
analyze_button = tk.Button(root, text="Analyze", command=analyze_text, font=("Helvetica", 12))
analyze_button.pack(pady=10)

clear_button = tk.Button(root, text="Clear All", command=clear_text, font=("Helvetica", 12))
clear_button.pack(pady=10)

root.mainloop()