import re
from youtube_transcript_api import YouTubeTranscriptApi
import torch
import gradio as gr
from transformers import pipeline

text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

# model_path = ("../Models/models--sshleifer--distilbart-cnn-12-6/snapshots"
#               "/a4f8f3ea906ed274767e9906dbaede7531d660ff")
# text_summary = pipeline("summarization", model=model_path,
#                 torch_dtype=torch.bfloat16)

def summary (input):
    output = text_summary(input)
    return output[0]['summary_text']

def extract_video_id(url):
    """
    Extracts YouTube video ID from URL.
    """
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None


def get_transcript_text(video_url):
    """
    Fetches transcript and returns as plain text.
    """
    video_id = extract_video_id(video_url)
    if not video_id:
        return "❌ Invalid YouTube URL."

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Convert list of dicts to single plain text
        full_text = " ".join([item['text'] for item in transcript])
        return full_text

    except Exception as e:
        return f"❌ Error fetching transcript: {e}"


# 🧪 Test
# if __name__ == "__main__":
#     url = "https://youtu.be/5PibknhIsTc"  # Replace with any valid YouTube URL
#     print(get_transcript_text(url))
gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text",outputs="text")
demo = gr.Interface(fn=get_transcript_text,
                    inputs=[gr.Textbox(label="Input YouTube Url to summarize",lines=1)],
                    outputs=[gr.Textbox(label="Summarized text",lines=4)],
                    title="@GenAILearniverse Project 2: YouTube Script Summarizer",
                    description="THIS APPLICATION WILL BE USED TO SUMMARIZE THE YOUTUBE VIDEO SCRIPT.")
demo.launch()
