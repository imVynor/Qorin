# ModernGPT-S 500M — SFT Canonical Loss-Mask Verification Example

This artifact proves exact token-level supervision and masking compliance according to the canonical ChatML protocol.

## 1. Input Conversation Messages
```json
[
  {
    "role": "system",
    "content": "You are ModernGPT-S, a helpful and precise AI assistant."
  },
  {
    "role": "user",
    "content": "Calculate the square of 17 using Python."
  },
  {
    "role": "assistant",
    "content": "<|tool_call|>{\"name\": \"python\", \"arguments\": {\"code\": \"print(17**2)\"}}"
  },
  {
    "role": "tool",
    "content": "<|tool_result|>{\"stdout\": \"289\\n\", \"exit_code\": 0}"
  },
  {
    "role": "assistant",
    "content": "The square of 17 is 289."
  }
]
```

## 2. Token-by-Token Masking Audit Table

| Position | Token ID | Token String | Label (Target) | Status | Role Turn Context |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 000 | `4` | `''` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 001 | `9410` | `'system'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 002 | `208` | `'\n'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 003 | `2987` | `'You'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 004 | `380` | `' are'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 005 | `7058` | `' Modern'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 006 | `48` | `'G'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 007 | `9117` | `'PT'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 008 | `22` | `'-'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 009 | `60` | `'S'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 010 | `21` | `','` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 011 | `267` | `' a'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 012 | `5603` | `' helpful'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 013 | `297` | `' and'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 014 | `8139` | `' precise'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 015 | `5586` | `' AI'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 016 | `12347` | `' assistant'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 017 | `23` | `'.'` | `-100` | MASKED (-100) | Header (<|im_start|>system) |
| 018 | `5` | `''` | `-100` | MASKED (-100) | Delimiter (<|im_end|>) |
| 019 | `208` | `'\n'` | `-100` | MASKED (-100) | Delimiter (<|im_end|>) |
| 020 | `4` | `''` | `-100` | MASKED (-100) | Header (<|im_start|>user) |
| 021 | `4599` | `'user'` | `-100` | MASKED (-100) | Header (<|im_start|>user) |
| 022 | `208` | `'\n'` | `-100` | MASKED (-100) | Header (<|im_start|>user) |
| 023 | `32240` | `'Calcul'` | `-100` | MASKED (-100) | Header (<|im_start|>user) |
| 024 | `382` | `'ate'` | `-100` | MASKED (-100) | Header (<|im_start|>user) |
| 025 | `273` | `' the'` | `-100` | MASKED (-100) | Header (<|im_start|>user) |
| 026 | `5366` | `' square'` | `-100` | MASKED (-100) | Header (<|im_start|>user) |
| 027 | `294` | `' of'` | `-100` | MASKED (-100) | Header (<|im_start|>user) |
| 028 | `2076` | `' 17'` | `-100` | MASKED (-100) | Header (<|im_start|>user) |
| 029 | `1019` | `' using'` | `-100` | MASKED (-100) | Header (<|im_start|>user) |
| 030 | `4920` | `' Python'` | `-100` | MASKED (-100) | Header (<|im_start|>user) |
| 031 | `23` | `'.'` | `-100` | MASKED (-100) | Header (<|im_start|>user) |
| 032 | `5` | `''` | `-100` | MASKED (-100) | Delimiter (<|im_end|>) |
| 033 | `208` | `'\n'` | `-100` | MASKED (-100) | Delimiter (<|im_end|>) |
| 034 | `4` | `''` | `-100` | MASKED (-100) | Header (<|im_start|>ass) |
| 035 | `526` | `'ass'` | `-100` | MASKED (-100) | Header (<|im_start|>ass) |
| 036 | `8647` | `'istant'` | `-100` | MASKED (-100) | Header (<|im_start|>ass) |
| 037 | `208` | `'\n'` | `-100` | MASKED (-100) | Header (<|im_start|>ass) |
| 038 | `6` | `''` | `6` | **SUPERVISED** | Header (<|im_start|>ass) |
| 039 | `100` | `'{'` | `100` | **SUPERVISED** | Header (<|im_start|>ass) |
| 040 | `11` | `'"'` | `11` | **SUPERVISED** | Header (<|im_start|>ass) |
| 041 | `1285` | `'name'` | `1285` | **SUPERVISED** | Header (<|im_start|>ass) |
| 042 | `3446` | `'":'` | `3446` | **SUPERVISED** | Header (<|im_start|>ass) |
| 043 | `531` | `' "'` | `531` | **SUPERVISED** | Header (<|im_start|>ass) |
| 044 | `4511` | `'python'` | `4511` | **SUPERVISED** | Header (<|im_start|>ass) |
| 045 | `1500` | `'",'` | `1500` | **SUPERVISED** | Header (<|im_start|>ass) |
| 046 | `531` | `' "'` | `531` | **SUPERVISED** | Header (<|im_start|>ass) |
| 047 | `990` | `'arg'` | `990` | **SUPERVISED** | Header (<|im_start|>ass) |
| 048 | `3008` | `'uments'` | `3008` | **SUPERVISED** | Header (<|im_start|>ass) |
| 049 | `3446` | `'":'` | `3446` | **SUPERVISED** | Header (<|im_start|>ass) |
| 050 | `19362` | `' {"'` | `19362` | **SUPERVISED** | Header (<|im_start|>ass) |
| 051 | `4838` | `'code'` | `4838` | **SUPERVISED** | Header (<|im_start|>ass) |
| 052 | `3446` | `'":'` | `3446` | **SUPERVISED** | Header (<|im_start|>ass) |
| 053 | `531` | `' "'` | `531` | **SUPERVISED** | Header (<|im_start|>ass) |
| 054 | `5185` | `'print'` | `5185` | **SUPERVISED** | Header (<|im_start|>ass) |
| 055 | `17` | `'('` | `17` | **SUPERVISED** | Header (<|im_start|>ass) |
| 056 | `2790` | `'17'` | `2790` | **SUPERVISED** | Header (<|im_start|>ass) |
| 057 | `900` | `'**'` | `900` | **SUPERVISED** | Header (<|im_start|>ass) |
| 058 | `27` | `'2'` | `27` | **SUPERVISED** | Header (<|im_start|>ass) |
| 059 | `24033` | `')"'` | `24033` | **SUPERVISED** | Header (<|im_start|>ass) |
| 060 | `14537` | `'}}'` | `14537` | **SUPERVISED** | Header (<|im_start|>ass) |
| 061 | `5` | `''` | `5` | **SUPERVISED** | Delimiter (<|im_end|>) |
| 062 | `208` | `'\n'` | `208` | **SUPERVISED** | Delimiter (<|im_end|>) |
| 063 | `4` | `''` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 064 | `13353` | `'tool'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 065 | `72` | `'_'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 066 | `18351` | `'obs'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 067 | `2887` | `'ervation'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 068 | `208` | `'\n'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 069 | `7` | `''` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 070 | `100` | `'{'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 071 | `11` | `'"'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 072 | `18788` | `'stdout'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 073 | `3446` | `'":'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 074 | `531` | `' "'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 075 | `4645` | `'28'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 076 | `34` | `'9'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 077 | `69` | `'\\'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 078 | `87` | `'n'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 079 | `1500` | `'",'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 080 | `531` | `' "'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 081 | `19966` | `'exit'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 082 | `72` | `'_'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 083 | `4838` | `'code'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 084 | `3446` | `'":'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 085 | `1325` | `' 0'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 086 | `102` | `'}'` | `-100` | MASKED (-100) | Header (<|im_start|>tool) |
| 087 | `5` | `''` | `-100` | MASKED (-100) | Delimiter (<|im_end|>) |
| 088 | `208` | `'\n'` | `-100` | MASKED (-100) | Delimiter (<|im_end|>) |
| 089 | `4` | `''` | `-100` | MASKED (-100) | Header (<|im_start|>ass) |
| 090 | `526` | `'ass'` | `-100` | MASKED (-100) | Header (<|im_start|>ass) |
| 091 | `8647` | `'istant'` | `-100` | MASKED (-100) | Header (<|im_start|>ass) |
| 092 | `208` | `'\n'` | `-100` | MASKED (-100) | Header (<|im_start|>ass) |
| 093 | `521` | `'The'` | `521` | **SUPERVISED** | Header (<|im_start|>ass) |
| 094 | `5366` | `' square'` | `5366` | **SUPERVISED** | Header (<|im_start|>ass) |
| 095 | `294` | `' of'` | `294` | **SUPERVISED** | Header (<|im_start|>ass) |
| 096 | `2076` | `' 17'` | `2076` | **SUPERVISED** | Header (<|im_start|>ass) |
| 097 | `333` | `' is'` | `333` | **SUPERVISED** | Header (<|im_start|>ass) |
| 098 | `4908` | `' 28'` | `4908` | **SUPERVISED** | Header (<|im_start|>ass) |
| 099 | `34` | `'9'` | `34` | **SUPERVISED** | Header (<|im_start|>ass) |
| 100 | `23` | `'.'` | `23` | **SUPERVISED** | Header (<|im_start|>ass) |
| 101 | `5` | `''` | `5` | **SUPERVISED** | Delimiter (<|im_end|>) |
| 102 | `208` | `'\n'` | `208` | **SUPERVISED** | Delimiter (<|im_end|>) |
| 103 | `9` | `''` | `9` | **SUPERVISED** | Sequence End (<|endoftext|>) |

## 3. Mathematical Verification Summary
- **Total Sequence Length:** 104 tokens
- **Supervised Tokens:** 36 tokens (34.6%)
- **Masked Tokens (-100):** 68 tokens
- **Integrity Rules Confirmed:**
  1. System and User turns have 100% tokens masked with `-100`.
  2. Tool observations (`<|tool_result|>`) have 100% tokens masked with `-100`.
  3. Assistant thoughts, text, tool calls (`<|tool_call|>`), and `<|im_end|>` are strictly supervised.
  4. Final `<|endoftext|>` is supervised to guarantee natural generation termination.