import sys
from pathlib import Path

sys.path.insert(0, ".")
from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper
from modern_gpt.data.conversation_serializer import ConversationSerializer, Message

tok = HuggingFaceBPEWrapper.load("modern_gpt/artifacts/tokenizer_16k_v2")
serializer = ConversationSerializer(tok)

messages = [
    {"role": "system", "content": "You are ModernGPT-S, a helpful and precise AI assistant."},
    {"role": "user", "content": "Hello"}
]

# Method A: chat.py prompt assembly
prompt = ""
for msg in messages:
    prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
prompt += "<|im_start|>assistant\n"
chat_ids = tok.encode(prompt)

# Method B: serializer piece-by-piece (as during training)
ser_ids = []
for msg in [Message(role=m["role"], content=m["content"]) for m in messages]:
    ser_ids.extend(tok.encode(f"<|im_start|>{msg.role}\n"))
    ser_ids.extend(tok.encode(msg.content))
    ser_ids.extend(tok.encode("<|im_end|>\n"))
ser_ids.extend(tok.encode("<|im_start|>assistant\n"))

print("Chat.py prompt text:\n", repr(prompt))
print("\nChat.py input_ids length:", len(chat_ids))
print("Serializer input_ids length:", len(ser_ids))
print("Identical token IDs?:", chat_ids == ser_ids)

if chat_ids != ser_ids:
    print("\nDifferences:")
    min_len = min(len(chat_ids), len(ser_ids))
    for idx in range(min_len):
        if chat_ids[idx] != ser_ids[idx]:
            print(f"Index {idx}: chat.py={chat_ids[idx]} ({repr(tok.decode([chat_ids[idx]]))}) vs serializer={ser_ids[idx]} ({repr(tok.decode([ser_ids[idx]]))})")
    if len(chat_ids) != len(ser_ids):
        print(f"Length mismatch: chat.py={len(chat_ids)} vs serializer={len(ser_ids)}")
