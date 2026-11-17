"""
Script to apply DEF-01, DEF-02, and DEF-03 repairs to backend/app.py
"""

from pathlib import Path

def apply_repairs():
    app_path = Path("backend/app.py")
    content = app_path.read_text(encoding="utf-8")

    start_marker = "# --- CHAT & SSE STREAMING ---"
    end_marker = "return EventSourceResponse(event_generator())\n"

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker) + len(end_marker)

    assert start_idx != -1 and end_idx != -1, "Markers not found in backend/app.py!"

    new_block = '''# --- CHAT & SSE STREAMING ---

def get_compact_tool_manifest(web_enabled: bool, python_enabled: bool) -> Optional[str]:
    """Generates the compact tool manifest strictly matching tool_context_spec.json."""
    if web_enabled and python_enabled:
        return (
            "You are ModernGPT-S, a helpful and precise AI assistant.\\n"
            "You have access to external tools:\\n"
            "1. `python`: Executes Python code in an isolated sandbox. Use for math, calculations, and data processing. Format: <|tool_call|>{\\"name\\": \\"python\\", \\"arguments\\": {\\"code\\": \\"<code_string>\\"}}\\n"
            "2. `web_research`: Searches the web for current factual information. Returns evidence with source URLs. Format: <|tool_call|>{\\"name\\": \\"web_research\\", \\"arguments\\": {\\"query\\": \\"<query_string>\\"}}\\n"
            "When a tool is needed, emit ONLY the tool call. If no tool is needed, respond directly to the user."
        )
    elif python_enabled:
        return (
            "You are ModernGPT-S, a helpful and precise AI assistant.\\n"
            "You have access to external tools:\\n"
            "1. `python`: Executes Python code in an isolated sandbox. Use for math, calculations, and data processing. Format: <|tool_call|>{\\"name\\": \\"python\\", \\"arguments\\": {\\"code\\": \\"<code_string>\\"}}\\n"
            "When a tool is needed, emit ONLY the tool call. If no tool is needed, respond directly to the user."
        )
    elif web_enabled:
        return (
            "You are ModernGPT-S, a helpful and precise AI assistant.\\n"
            "You have access to external tools:\\n"
            "1. `web_research`: Searches the web for current factual information. Returns evidence with source URLs. Format: <|tool_call|>{\\"name\\": \\"web_research\\", \\"arguments\\": {\\"query\\": \\"<query_string>\\"}}\\n"
            "When a tool is needed, emit ONLY the tool call. If no tool is needed, respond directly to the user."
        )
    return None


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Server-Sent Events (SSE) streaming endpoint with closed-loop tool augmentation."""
    trace_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    raw_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    last_user_msg = next((m["content"] for m in reversed(raw_messages) if m["role"] == "user"), "")

    async def event_generator():
        # 1. Step: Retrieval Evaluation
        retrieval_trace = retrieval_engine.retrieve(
            query=last_user_msg,
            mode=req.retrieval_mode,
            top_k=req.retrieval_top_k,
            threshold=req.retrieval_threshold,
        )

        yield {
            "event": "retrieval",
            "data": json.dumps(retrieval_trace.to_dict()),
        }

        # Inject context into messages if triggered
        generation_messages = retrieval_engine.inject_context_into_messages(
            raw_messages, retrieval_trace
        )

        # 2. Step: Compact Tool Manifest Injection (DEF-02) & Autonomous Routing (DEF-03)
        compact_manifest = get_compact_tool_manifest(req.web_research_enabled, req.python_sandbox_enabled)
        if req.system_prompt and compact_manifest and compact_manifest not in req.system_prompt:
            effective_system_prompt = f"{compact_manifest}\\n\\n{req.system_prompt}"
        else:
            effective_system_prompt = req.system_prompt or compact_manifest

        config = GenerationConfig(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_new_tokens=req.max_new_tokens,
            repetition_penalty=req.repetition_penalty,
            system_prompt=effective_system_prompt,
        )

        # 3. Step: Closed-Loop Model Generation Stream (DEF-01)
        tool_traces = []
        active_messages = list(generation_messages)
        final_response_text = ""
        ttft_recorded = 0.0
        final_tok_per_sec = 0.0
        final_token_count = 0

        max_tool_rounds = 2
        current_round = 0

        try:
            while current_round <= max_tool_rounds:
                round_response_text = ""
                round_chunks = []

                async for chunk in model_manager.generate_stream(req.model_id, active_messages, config):
                    round_response_text += chunk.text
                    round_chunks.append(chunk)

                    if chunk.ttft_ms and ttft_recorded == 0.0:
                        ttft_recorded = chunk.ttft_ms
                    if chunk.tok_per_sec:
                        final_tok_per_sec = chunk.tok_per_sec
                    if chunk.token_count:
                        final_token_count += chunk.token_count

                # Check for tool call in output
                is_tool_call = ("<tool_call>" in round_response_text or "<|tool_call|>" in round_response_text)

                if is_tool_call and current_round < max_tool_rounds and (req.web_research_enabled or req.python_sandbox_enabled):
                    parsed_tool = tool_orchestrator.parse_model_tool_call(round_response_text)
                    if parsed_tool:
                        tname, targs = parsed_tool
                        yield {
                            "event": "tool_start",
                            "data": json.dumps({"tool_name": tname, "status": "started", "args": targs}),
                        }
                        t_trace = tool_orchestrator.execute_tool(
                            tname,
                            targs,
                            requested_by="model",
                            web_enabled=req.web_research_enabled,
                            python_enabled=req.python_sandbox_enabled,
                        )
                        tool_traces.append(t_trace.to_dict())
                        yield {
                            "event": "tool_result",
                            "data": json.dumps(t_trace.to_dict()),
                        }

                        # Append assistant tool call and tool observation to active_messages for closed loop
                        active_messages.append({"role": "assistant", "content": round_response_text})
                        obs_tag = "<|tool_result|>" if t_trace.status == "completed" else "<|tool_error|>"
                        obs_payload = t_trace.output_payload
                        active_messages.append({
                            "role": "tool",
                            "content": f"{obs_tag}{json.dumps(obs_payload)}",
                        })

                        current_round += 1
                        continue
                    else:
                        # Malformed tool call: report error and allow bounded recovery
                        yield {
                            "event": "tool_result",
                            "data": json.dumps({"status": "failed", "error": "Malformed tool call syntax rejected."}),
                        }
                        active_messages.append({"role": "assistant", "content": round_response_text})
                        err_payload = json.dumps({"error": "Malformed tool call syntax rejected."})
                        active_messages.append({
                            "role": "tool",
                            "content": f"<|tool_error|>{err_payload}",
                        })
                        current_round += 1
                        continue
                else:
                    # Direct answer or second-hop synthesized response: stream tokens to client
                    final_response_text = round_response_text
                    for c in round_chunks:
                        yield {
                            "event": "token",
                            "data": json.dumps({
                                "text": c.text,
                                "is_finished": c.is_finished,
                                "token_count": c.token_count,
                                "ttft_ms": round(c.ttft_ms, 1) if c.ttft_ms else None,
                                "tok_per_sec": round(c.tok_per_sec, 1) if c.tok_per_sec else None,
                                "elapsed_ms": round(c.elapsed_ms, 1) if c.elapsed_ms else None,
                            }),
                        }
                    break

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }
            return

        total_elapsed = (time.perf_counter() - start_time) * 1000.0

        # 4. Step: Save Structured Experiment Trace
        meta = model_manager.get_model_metadata(req.model_id)
        exp_trace = ExperimentTrace(
            id=trace_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_id=req.model_id,
            provider=meta.provider if meta else "unknown",
            checkpoint=meta.checkpoint_path or "N/A" if meta else "N/A",
            prompt=last_user_msg,
            messages=raw_messages,
            generation_config=config.__dict__,
            retrieval_trace=retrieval_trace.to_dict() if retrieval_trace.triggered else None,
            tool_traces=tool_traces,
            response_text=final_response_text,
            token_count=final_token_count,
            ttft_ms=round(ttft_recorded, 2),
            total_latency_ms=round(total_elapsed, 2),
            tokens_per_second=round(final_tok_per_sec, 2),
        )
        experiment_tracker.log_experiment(exp_trace)

        yield {
            "event": "done",
            "data": json.dumps({
                "trace_id": trace_id,
                "total_tokens": final_token_count,
                "ttft_ms": round(ttft_recorded, 2),
                "total_latency_ms": round(total_elapsed, 2),
                "tok_per_sec": round(final_tok_per_sec, 2),
                "retrieval_triggered": retrieval_trace.triggered,
                "tools_executed": len(tool_traces),
            }),
        }

    return EventSourceResponse(event_generator())
'''

    updated_content = content[:start_idx] + new_block + content[end_idx:]
    app_path.write_text(updated_content, encoding="utf-8")
    print("Successfully patched backend/app.py!")

if __name__ == "__main__":
    apply_repairs()
