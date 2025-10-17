chat_doc = {
    'tags': ['Chat'],
    'summary': 'Send a message to a selected model and receive a response',
    'description': 'This endpoint sends a user message to the specified model and returns the generated reply (e.g., TinyLlama, Qwen, GPT-2, Deepseek, Redhat).',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {
                        'type': 'string',
                    },
                    'model': {
                        'type': 'string',
                        'default': 'TinyLlama'
                    }
                },
                'required': ['message']
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Model response generated successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'response': {
                        'type': 'string',
                    }
                }
            }
        },
        400: {
            'description': 'Model not found or invalid input'
        }
    }
}