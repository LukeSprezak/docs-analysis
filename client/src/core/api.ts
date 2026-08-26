
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1';

export interface DocumentInfo {
  id: string;
  filename: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

export interface ConversationInfo {
  id: string;
  title: string;
  messages: ChatMessage[];
  created_at?: string;
}

export interface SummaryInfo {
  id: string;
  summary: string;
  document_ids: string[];
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  email: string;
}

const TOKEN_KEY = 'authToken';

export const tokenStorage = {
  get: (): string | null => localStorage.getItem(TOKEN_KEY),
  set: (token: string): void => localStorage.setItem(TOKEN_KEY, token),
  clear: (): void => localStorage.removeItem(TOKEN_KEY),
};

let onUnauthorized: (() => void) | null = null;

export const setUnauthorizedHandler = (handler: (() => void) | null): void => {
  onUnauthorized = handler;
};

const handleUnauthorized = (): void => {
  tokenStorage.clear();
  if (onUnauthorized) onUnauthorized();
};

const authHeader = (): Record<string, string> => {
  const token = tokenStorage.get();
  return token ? { Authorization: `Bearer ${token}` } : {};
};

const errorCodeFromBody = (body: string): string => {
  try {
    return JSON.parse(body)?.error?.error_code || 'GENERIC';
  } catch {
    return 'GENERIC';
  }
};

const extractErrorCode = async (response: Response): Promise<string> =>
  errorCodeFromBody(await response.text());

const ensureOk = async (response: Response): Promise<Response> => {
  if (!response.ok) {
    throw new Error(await extractErrorCode(response));
  }
  return response;
};

const authedFetch = async (url: string, options: RequestInit = {}): Promise<Response> => {
  const response = await fetch(url, {
    ...options,
    headers: { ...(options.headers || {}), ...authHeader() },
  });

  if (response.status === 401) {
    handleUnauthorized();
    throw new Error('UNAUTHORIZED');
  }
  return ensureOk(response);
};

export const api = {
  register: async (email: string, password: string): Promise<AuthResponse> => {
    const response = await fetch(`${API_BASE_URL}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    return (await ensureOk(response)).json();
  },
  login: async (email: string, password: string): Promise<AuthResponse> => {
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    return (await ensureOk(response)).json();
  },
  uploadDocument: async (file: File, onProgress?: (percent: number) => void): Promise<any> => {
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append('file', file);

      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${API_BASE_URL}/documents/upload`);

      const token = tokenStorage.get();
      if (token) {
        xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      }

      if (onProgress) {
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const percentComplete = (event.loaded / event.total) * 100;
            onProgress(percentComplete);
          }
        };
      }

      xhr.onload = () => {
        if (xhr.status === 401) {
          handleUnauthorized();
          reject(new Error('UNAUTHORIZED'));
          return;
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            resolve(xhr.responseText);
          }
        } else {
          reject(new Error(errorCodeFromBody(xhr.responseText)));
        }
      };

      xhr.onerror = () => reject(new Error('Network error during upload'));
      xhr.send(formData);
    });
  },
  listDocuments: async (): Promise<DocumentInfo[]> => {
    const response = await authedFetch(`${API_BASE_URL}/documents/`);
    return response.json();
  },
  deleteDocument: async (id: string): Promise<void> => {
    await authedFetch(`${API_BASE_URL}/documents/${encodeURIComponent(id)}`, { method: 'DELETE' });
  },
  chat: async (message: string, history: ChatMessage[], conversation_id?: string) => {
    const response = await authedFetch(`${API_BASE_URL}/chat/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message, history, conversation_id }),
    });
    return response.json();
  },
  chatStream: async (
    message: string,
    history: ChatMessage[],
    conversation_id: string | undefined,
    onToken: (token: string) => void,
    onDone: (data: { conversation_id: string; sources: string[] }) => void,
  ): Promise<void> => {
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authHeader(),
      },
      body: JSON.stringify({ message, history, conversation_id }),
    });

    if (response.status === 401) {
      handleUnauthorized();
      throw new Error('UNAUTHORIZED');
    }
    if (!response.ok || !response.body) {
      throw new Error(`Chat stream failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const handleLine = (line: string) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      const event = JSON.parse(trimmed);
      if (event.type === 'token') {
        onToken(event.content);
      } else if (event.type === 'done') {
        onDone({ conversation_id: event.conversation_id, sources: event.sources });
      }
    };

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      lines.forEach(handleLine);
    }
    handleLine(buffer);
  },
  listConversations: async (): Promise<ConversationInfo[]> => {
    const response = await authedFetch(`${API_BASE_URL}/chat/conversations`);
    return response.json();
  },
  getConversation: async (id: string): Promise<ConversationInfo> => {
    const response = await authedFetch(`${API_BASE_URL}/chat/conversations/${encodeURIComponent(id)}`);
    return response.json();
  },
  deleteConversation: async (id: string): Promise<void> => {
    await authedFetch(`${API_BASE_URL}/chat/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' });
  },
  askQuestion: async (question: string) => {
    const response = await authedFetch(`${API_BASE_URL}/qa/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });
    return response.json();
  },
  summarize: async (document_ids: string[]): Promise<SummaryInfo> => {
    const response = await authedFetch(`${API_BASE_URL}/summarize/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ document_ids }),
    });
    return response.json();
  },
  listSummaries: async (): Promise<SummaryInfo[]> => {
    const response = await authedFetch(`${API_BASE_URL}/summarize/`);
    return response.json();
  },
  deleteSummary: async (id: string): Promise<void> => {
    await authedFetch(`${API_BASE_URL}/summarize/${encodeURIComponent(id)}`, { method: 'DELETE' });
  },
  getTranslations: async (lang: string): Promise<Record<string, string>> => {
    const response = await fetch(`${API_BASE_URL}/translations/${encodeURIComponent(lang)}`);
    return (await ensureOk(response)).json();
  },
};
