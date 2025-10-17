upload_pdf_doc = {
    'tags': ['PDF'],
    'summary': 'Upload and scan a PDF file',
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'pdf',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'PDF file to upload'
        }
    ],
    'responses': {
        200: {
            'description': 'PDF scanned successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'filename': {'type': 'string'}
                }
            }
        },
        400: {'description': 'No file uploaded'}
    }
}

ask_pdf_doc = {
    'tags': ['PDF'],
    'summary': 'Ask a question based on the uploaded PDF (e.g., TinyLlama, Qwen, GPT-2, Deepseek, Redhat).',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'question': {'type': 'string'},
                    'model': {'type': 'string'}
                }
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Answer generated from PDF',
            'schema': {
                'type': 'object',
                'properties': {
                    'answer': {'type': 'string'}
                }
            }
        },
        400: {'description': 'Unsupported model or bad request'},
        404: {'description': 'No PDF found'}
    }
}

remove_pdf_doc = {
    'tags': ['PDF'],
    'summary': 'Remove the uploaded PDF and clear the database',
    'responses': {
        200: {
            'description': 'PDF removed',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'}
                }
            }
        }
    }
}