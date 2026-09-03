curl -N -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-4B","stream":true,
       "messages":[{"role":"user","content":"How many blades of grass are on all the golf courses in Scotland? Reason step by step, then end with ANSWER: <number>."}],
       "max_tokens":2048,"temperature":1.0,"top_p":0.95,"presence_penalty":1.5}' \
| sed -u 's/^data: //' | grep -v '^\[DONE\]' \
| jq -j --unbuffered '.choices[0].delta.content // empty'
