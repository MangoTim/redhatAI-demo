history_doc = {
    'tags': ['History'],
    'summary': 'Retrieve recent chat history',
    'description': 'Returns the last 10 messages exchanged between the user and assistant, ordered by timestamp.',
    'responses': {
        200: {
            'description': 'List of recent 10 chat messages',
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'role': {
                            'type': 'string',
                        },
                        'message': {
                            'type': 'string',
                        }
                    }
                }
            }
        }
    }
}