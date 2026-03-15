import json
import asyncio
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dataclasses import dataclass, field

# ── Config ────────────────────────────────────────────────────
with open("config.json") as f:
    CONFIG = json.load(f)

SYSTEM_PROMPT  = CONFIG["system_prompt"]
PATIENT        = CONFIG["patient"]
COLLECT_FIELDS = CONFIG["collect_fields"]

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "llama3.2:latest"

app = FastAPI()

print("Server started. Open http://localhost:8000 in your browser to test the demo.")

# ─────────────────────────────────────────────────────────────
# Session state
#
# Maya already knows the patient. She is collecting 4 fields
# from the rep. State only advances when tool_store confirms
# a value — never based on what the LLM says alone.
# ─────────────────────────────────────────────────────────────
@dataclass
class Session:
    history:   list = field(default_factory=list)
    collected: dict = field(default_factory=dict)   # confirmed fields
    call_done: bool = False

    def push(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def next_field(self) -> dict | None:
        for f in COLLECT_FIELDS:
            if f["key"] not in self.collected:
                return f
        return None

    def all_collected(self) -> bool:
        return all(f["key"] in self.collected for f in COLLECT_FIELDS)


# ─────────────────────────────────────────────────────────────
# Tool — the only way state advances
# 

# tool_store_field: called after rep provides a value.
# Until this succeeds, session.collected does not update.
# ─────────────────────────────────────────────────────────────
async def tool_store_field(key: str, value: str) -> str:
    """
    Simulate writing to a claims system.
    Replace body with real DB/API write in production.
    Returns stored value on success, 'INVALID' on failure.
    """
    await asyncio.sleep(1)
    cleaned = value.strip()
    if not cleaned:
        return "INVALID"
    # Reject if it looks like a failed extraction slipped through
    normalised = cleaned.upper().replace(" ", "").replace("_", "").replace("-", "")
    if normalised in ("NOTFOUND", "NA", "NONE", "NULL", "NOTAVAILABLE", "UNKNOWN"):
        return "INVALID"
    return cleaned


# ─────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────
async def llm_stream(messages: list, ws: WebSocket) -> str:
    full = ""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", OLLAMA_URL,
            json={"model": MODEL, "messages": messages, "stream": True}
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "message" in chunk:
                    token = chunk["message"]["content"]
                    full += token
                    await ws.send_text(json.dumps({"type": "token", "content": token}))
                if chunk.get("done"):
                    break
    return full


async def llm_single(messages: list) -> str:
    full = ""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", OLLAMA_URL,
            json={"model": MODEL, "messages": messages, "stream": True}
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "message" in chunk:
                    full += chunk["message"]["content"]
                if chunk.get("done"):
                    break
    return full.strip()


# ─────────────────────────────────────────────────────────────
# Extract a specific field value from the rep's message
# ─────────────────────────────────────────────────────────────
async def extract_value(label: str, rep_msg: str, other_labels: list[str]) -> str | None:
    others = ", ".join(other_labels) if other_labels else "none"
    prompt = (
        f"Task: extract ONLY the '{label}' from the message below.\n"
        f"Rules:\n"
        f"  - Reply with ONLY the raw value for '{label}', nothing else.\n"
        f"  - The value must be explicitly stated AND clearly intended as the '{label}'.\n"
        f"  - Do NOT extract values that belong to other fields such as: {others}.\n"
        f"  - Do NOT infer, guess, or match partial content.\n"
        f"  - If the '{label}' is not clearly and explicitly present, reply: NOTFOUND\n\n"
        f"Message: {rep_msg}\n\n"
        f"What is the '{label}'? Reply NOTFOUND if not present."
    )
    result = await llm_single([{"role": "user", "content": prompt}])
    result = result.strip()
    normalised = result.upper().replace(" ", "").replace("_", "").replace("-", "")
    if not result or normalised in ("NOTFOUND", "NA", "NONE", "NULL", ""):
        return None
    return result


# ─────────────────────────────────────────────────────────────
# Build system prompt — state injected every turn
# ─────────────────────────────────────────────────────────────
def build_system(session: Session, situation: str = "") -> str:
    p = PATIENT
    lines = [
        SYSTEM_PROMPT, "",
        "PATIENT ON FILE:",
        f"  Name:      {p['name']}",
        f"  DOB:       {p['dob']}",
        f"  Member ID: {p['member_id']}",
        f"  Tax ID:    {p['tax_id']}",
        f"  NPI ID:    {p['npi_id']}",
        "",
        "Provide any of the above details if the rep asks to verify the patient.",
        "",
        "COLLECTED SO FAR:",
    ]

    for f in COLLECT_FIELDS:
        val = session.collected.get(f["key"], "NOT YET COLLECTED")
        lines.append(f"  - {f['label']}: {val}")

    nxt = session.next_field()
    if nxt:
        ask = nxt["ask"].format(patient_name=p["name"])
        lines += [
            "",
            f"YOUR NEXT TASK: Collect the '{nxt['label']}'.",
            f"Ask the rep: \"{ask}\"",
            "Do NOT move to the next field until this one is confirmed.",
            "Do NOT close the call. Do NOT say goodbye.",
            "If the rep gives unrelated information or tries to end the call,",
            "acknowledge briefly and re-ask for this specific field.",
        ]
    else:
        lines += [
            "",
            "ALL FIELDS COLLECTED. Thank the rep, summarise the call details, and close the call.",
        ]

    if situation:
        lines += ["", f"CURRENT SITUATION: {situation}"]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# State push to frontend
# ─────────────────────────────────────────────────────────────
async def push_state(ws: WebSocket, session: Session):
    await ws.send_text(json.dumps({
        "type":         "state",
        "collected":    session.collected,
        "collect_fields": [f["key"] for f in COLLECT_FIELDS],
        "field_labels": {f["key"]: f["label"] for f in COLLECT_FIELDS},
        "call_done":    session.call_done,
    }))


# ─────────────────────────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def chat(ws: WebSocket):
    await ws.accept()
    session = Session()

    await push_state(ws, session)

    try:
        # ── Maya opens the call — no input needed ─────────────
        opening = await llm_stream([{
            "role": "system",
            "content": build_system(session, situation=(
                "This is the start of the call. Introduce yourself as Maya from "
                "Northwell Health's claims department. State you are calling to check "
                "on a claim for the patient on file. Ask the rep to verify the member's identity."
            ))
        }], ws)
        session.push("assistant", opening)
        await ws.send_text(json.dumps({"type": "done"}))

        # ── Main loop ─────────────────────────────────────────
        while not session.call_done:
            raw     = await ws.receive_text()
            rep_msg = json.loads(raw).get("message", "").strip()
            if not rep_msg:
                continue

            session.push("user", rep_msg)
            nxt = session.next_field()

            if not nxt:
                # All collected — shouldn't normally reach here, but handle gracefully
                session.call_done = True
                await push_state(ws, session)
                await ws.send_text(json.dumps({"type": "call_complete"}))
                await ws.send_text(json.dumps({"type": "done"}))
                continue

            # ── Try to extract the expected field ─────────────
            # Pass all OTHER field labels so the extractor doesn't grab the wrong value
            other_labels = [f["label"] for f in COLLECT_FIELDS if f["key"] != nxt["key"]]
            extracted = await extract_value(nxt["label"], rep_msg, other_labels)

            if extracted:
                # ── Tool call: store the value concurrently ───
                # Stream acknowledgement while tool writes to system
                await ws.send_text(json.dumps({
                    "type":  "tool_start",
                    "label": f"Recording {nxt['label']}…"
                }))

                bridge_msgs = [
                    {"role": "system", "content": build_system(session, situation=(
                        f"The rep just provided the {nxt['label']}: '{extracted}'. "
                        f"Acknowledge it warmly in one short sentence while you record it."
                    ))}
                ] + session.history

                # Concurrent: LLM streams acknowledgement, tool stores value
                stream_task = asyncio.create_task(llm_stream(bridge_msgs, ws))
                store_task  = asyncio.create_task(tool_store_field(nxt["key"], extracted))

                bridge_reply, stored = await asyncio.gather(stream_task, store_task)

                await ws.send_text(json.dumps({"type": "tool_done"}))
                session.push("assistant", bridge_reply)

                if stored != "INVALID":
                    # ── State advances HERE — tool confirmed ──
                    session.collected[nxt["key"]] = stored
                    await push_state(ws, session)

                    if session.all_collected():
                        # Close the call
                        session.call_done = True
                        close_reply = await llm_stream([
                            {"role": "system", "content": build_system(session)}
                        ] + session.history, ws)
                        session.push("assistant", close_reply)
                        await ws.send_text(json.dumps({"type": "call_complete"}))
                    else:
                        # Ask for next field
                        next_f = session.next_field()
                        ask_reply = await llm_stream([
                            {"role": "system", "content": build_system(
                                session,
                                situation=f"Good. '{nxt['label']}' has been recorded. Now ask for the {next_f['label']}."
                            )}
                        ] + session.history, ws)
                        session.push("assistant", ask_reply)
                else:
                    # Store failed — re-ask
                    retry_reply = await llm_stream([
                        {"role": "system", "content": build_system(session, situation=(
                            f"The value provided for {nxt['label']} was not valid. "
                            f"Politely ask the rep to provide it again."
                        ))}
                    ] + session.history, ws)
                    session.push("assistant", retry_reply)

            else:
                # ── Rep did not provide the expected field ─────
                # State does NOT advance. Maya re-asks.
                redirect_reply = await llm_stream([
                    {"role": "system", "content": build_system(session, situation=(
                        f"The rep's message did not contain the {nxt['label']}. "
                        f"They said: '{rep_msg}'. "
                        f"Acknowledge what they said, then re-ask for the {nxt['label']}."
                    ))}
                ] + session.history, ws)
                session.push("assistant", redirect_reply)

            await push_state(ws, session)
            await ws.send_text(json.dumps({"type": "done"}))

    except WebSocketDisconnect:
        pass


# ─────────────────────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────────────────────
@app.get("/")
async def home():
    with open("index.html") as f:
        return HTMLResponse(f.read())