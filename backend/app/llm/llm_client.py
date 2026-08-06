"""
File: llm_client.py

Purpose:
Uses a local Llama model through Ollama to convert
recognized ASL signs into natural English.
"""
import ollama

class LLMTranslator:
    def __init__(self):
        self.model="llama3"

    def translate(self, signs):
        prompt=f"""
You are an ASL interpreter.

Convert the following recognized ASL signs into
natural, grammatically correct English.

Rules:
- Preserve the meaning.
- Do NOT invent information.
- Do NOT add extra words.
- Only improve grammar.

Recognized signs:
{" ".join(signs)} this joins the gestures together
"""
        response=ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",#each message comes from a user
                    "content": prompt
                }
            ]
        )

        return response.message.content.strip()
if __name__=="__main__": 
    translator=LLMTranslator()
    result=translator.translate([
        "HELLO",
        "YES"
    ])

    print(result)