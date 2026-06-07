
const API_BASE_URL = process.env.REACT_APP_API_URL || '/api/v1';

export interface DocumentInfo {
  id: string;
  filename: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export const api = {
  uploadDocument: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}/documents/upload`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  },
  listDocuments: async (): Promise<DocumentInfo[]> => {
    const response = await fetch(`${API_BASE_URL}/documents/`);
    return response.json();
  },
  chat: async (message: string, history: ChatMessage[]) => {
    const response = await fetch(`${API_BASE_URL}/chat/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message, history }),
    });
    return response.json();
  },
  askQuestion: async (question: string) => {
    const response = await fetch(`${API_BASE_URL}/qa/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });
    return response.json();
  },
  summarize: async (document_ids: string[]) => {
    const response = await fetch(`${API_BASE_URL}/summarize/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ document_ids }),
    });
    return response.json();
  },
};
