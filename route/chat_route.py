from flask import request, jsonify
from modules import get_model, get_context_from_keywords, log_message, clean_reply, get_db_connection, remove_speculative_intro, load_prompt_template, clean_model_output, extract_answer

from flasgger import swag_from
from api_docs.chat_route_docs import chat_doc

def chat_route(app):
    @app.route("/chat", methods=["POST"])
    @swag_from(chat_doc)
    def chat():
        data = request.get_json()
        user_message = data.get("message", "")
        model_key = data.get("model", "TinyLlama")
        max_length = 256
        max_tokens = 2048 - max_length

        agent, is_chat_model, has_tokenizer = get_model(model_key)
        print("The model:", model_key, flush=True)

        if agent is None:
            return jsonify({"error": f"Model '{model_key}' not found"}), 400

        log_message("user", user_message)
        context = get_context_from_keywords(user_message)
        prompt_template = load_prompt_template()

        # Handle direct keyword match
        if context and len(context.split("\n")) == 1:
            keyword, response = context.split(": ", 1)
            print(f"Direct match found for keyword: {keyword}", flush=True)
            response = response.strip()
            if response.lower().startswith(keyword.lower()):
                response = response[len(keyword):].strip(" :.-")
            log_message("assistant", response)
            return jsonify({"response": response})

        # Build prompt
        if not context:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ]
            prompt = (
                agent.tokenizer.apply_chat_template(messages, tokenize=False)
                if is_chat_model and has_tokenizer else user_message
            )
        else:
            formatted_prompt = prompt_template.format(question=user_message)
            prompt_text = prompt_template.format(context=context or "")
            messages = [
                {"role": "system", "content": "You are a professional assistant."},
                {"role": "user", "content": formatted_prompt}
            ]
            if has_tokenizer:
                if is_chat_model:
                    prompt = agent.tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    encoded = agent.tokenizer.encode(prompt_text, max_length=max_length, truncation=True)
                    prompt = agent.tokenizer.decode(encoded)
            else:
                prompt = prompt_text

        print("Final prompt sent to model:\n", prompt, flush=True)

        # Generate response
        outputs = agent(prompt, max_new_tokens=max_tokens, do_sample=False, temperature=0.7, top_k=0, top_p=1.0)
        raw_output = outputs[0]["generated_text"]
        print("Raw output text:\n", raw_output, flush=True)

        # Clean and return reply
        reply = extract_answer(raw_output)
        reply = remove_speculative_intro(reply)
        answer = clean_model_output(reply, prompt_text=prompt, similarity_threshold=0.9)
        answer = clean_reply(answer, user_message)
        log_message("assistant", answer)

        print("Assistant reply content:\n", answer, flush=True)
        return jsonify({"response": answer})