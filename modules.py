import psycopg2
import re
from model.llama_agent import LlamaAgent
from model.gpt_agent import GPTAgent
from model.qwen_agent import QwenAgent
from model.redhat_test import RedhatAgent
from model.deepseek_agent import DeepSeekAgent
from model.qwen3_agent import Qwen3Agent
from difflib import SequenceMatcher

model_cache = {}

def get_db_connection():
    return psycopg2.connect(
        dbname="rhai_table",
        user="rhai",
        password="redhat",
        host="192.168.147.103",
        port=5432
    )

def get_model(model_key):
    key = model_key.strip().lower()

    if key == "tinyllama":
        if "tinyllama" not in model_cache:
            model_cache["tinyllama"] = LlamaAgent()
        return model_cache["tinyllama"], True, True

    elif key == "gpt-2":
        if "gpt-2" not in model_cache:
            model_cache["gpt-2"] = GPTAgent()
        return model_cache["gpt-2"], False, True

    elif key == "qwen":
        if "qwen" not in model_cache:
            model_cache["qwen"] = QwenAgent()
        return model_cache["qwen"], True, True

    elif key == "deepseek":
        if "deepseek" not in model_cache:
            model_cache["deepseek"] = DeepSeekAgent()
        return model_cache["deepseek"], True, False

    elif key == "redhat":
        if "redhat" not in model_cache:
            model_cache["redhat"] = RedhatAgent()
        return model_cache["redhat"], True, False

    elif key == "qwen3":
        if "qwen3" not in model_cache:
            model_cache["qwen3"] = Qwen3Agent()
        return model_cache["qwen3"], True, False

    print(f"Unknown model key: {model_key}", flush=True)
    return None, False, False


def get_context_from_keywords(message):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT keyword, response FROM keyword_responses")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        message_lower = message.lower()
        context_parts = []
        matched_keywords = []

        rows.sort(key=lambda x: len(x[0]), reverse=True)

        for keyword, response in rows:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, message_lower):
                context_parts.append(f"{keyword}: {response}")
                matched_keywords.append(keyword)

        if not matched_keywords:
            print("No keywords matched for this message.", flush=True)

        print("Matched keywords:", matched_keywords, flush=True)
        print("Injected context from DB:\n", "\n".join(context_parts), flush=True)

        return "\n".join(context_parts) if context_parts else ""

    except Exception as e:
        print(f"Error in keyword matching: {e}", flush=True)
        return ""

def log_message(role, message):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO chat_history (role, message) VALUES (%s, %s)", (role, message))
    conn.commit()
    cur.execute("""
        DELETE FROM chat_history
        WHERE id NOT IN (
            SELECT id FROM (
                SELECT id
                FROM chat_history
                WHERE role IN ('user', 'assistant')
                ORDER BY timestamp DESC
                LIMIT 10
            ) AS latest_pairs
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def clean_reply(reply, user_message):
    for tag in [
        "<|im_start|>", "<|im_end|>", "<||>", "〈||〉", "〈｜｜〉", "</s>", "**", "<|user|>", "<|assistant|>",
        "system:", "user:", "assistant:", "header:", "Chat History:", "User:", "Assistant:", "System:", "User:", "Assistant:", "System:"
    ]:
        reply = reply.replace(tag, "")
    if len(user_message.strip()) > 5 and user_message.strip().lower() in reply.strip().lower():
        reply = reply.lower().split(user_message.strip().lower())[-1].strip()
    return reply.strip()

def load_prompt_template(file_path="prompt.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf_prompt_template(file_path="prompt_pdf.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def clean_model_output(text, prompt_text=None, similarity_threshold=0.9):
    if prompt_text and text.startswith(prompt_text):
        text = text[len(prompt_text):].strip()

    preface_patterns = [
        r"^(let me (start|think)|i'm (planning|wondering)|we (need|should)|first, let's|okay, so|what should i|can i|could i|should i)\b.*?[.?!]",
        r"^(is it possible|do you know|maybe|perhaps|i think|i believe|i guess)\b.*?[.?!]",
        r"^(what are the best|what are some good|what activities can i do|what should i pack)\b.*?[.?!]",
    ]
    for pattern in preface_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()

    paragraphs = re.split(r'\n{2,}', text.strip())
    para_clean = []
    para_seen = []

    for para in paragraphs:
        normalized = para.strip().lower()
        is_duplicate = any(SequenceMatcher(None, normalized, prev).ratio() > similarity_threshold for prev in para_seen)
        if not is_duplicate:
            para_clean.append(para.strip())
            para_seen.append(normalized)

    final_lines = []
    line_seen = set()

    for para in para_clean:
        lines = para.splitlines()
        unique_lines = []
        for line in lines:
            normalized_line = line.strip().lower()
            if normalized_line not in line_seen:
                unique_lines.append(line.strip())
                line_seen.add(normalized_line)
        final_lines.append("\n".join(unique_lines))

    return "\n\n".join(final_lines)

def remove_speculative_intro(text):
    lines = text.split("\n")
    filtered = []
    for line in lines:
        if any(phrase in line.lower() for phrase in [
            "i'm planning", "let me start", "we need to", "i need to figure out",
            "i'm wondering", "we should consider", "let me recall", "first, let's"
        ]):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()

def extract_answer(raw_output: str) -> str:
    """
    Extracts the final answer from model output using known answer markers or JSON.
    Falls back to full output if no marker or answer field is found.
    """
    if not raw_output:
        return ""

    markers = [
        "Answer:",
        "Final Answer:",
        "Therefore, the answer is:",
        "The answer is:",
        "So the answer is:",
        "In conclusion:",
        "the answer is:",
        "答案:"

    ]
    pattern = "|".join(re.escape(marker) for marker in markers)
    match = re.search(pattern, raw_output)
    if match:
        start = match.end()
        answer = raw_output[start:].strip()
        answer = re.split(pattern, answer)[0].strip()
        return answer

    json_match = re.search(r"\{.*\"answer\"\s*:\s*\".*?\"\s*\}", raw_output, re.DOTALL)
    if json_match:
        try:
            answer_obj = json.loads(json_match.group())
            if "answer" in answer_obj:
                return answer_obj["answer"].strip()
        except json.JSONDecodeError:
            pass
    # Fallback to full output
    return raw_output.strip()

