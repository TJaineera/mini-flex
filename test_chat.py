from mini_flex.llms.openai_like import OpenAILike

llm = OpenAILike()
out = llm.chat([
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Say hello in 5 words."}
])
print(out)
