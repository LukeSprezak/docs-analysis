/**
 * Tests for the data layer.
 *
 * `api.ts` is the only place the client talks to the backend, so every response-shape
 * assumption lives here: the bearer token, the 401 path that logs the user out, the error
 * code dug out of the error envelope, and the hand-rolled NDJSON reader that carries the
 * whole chat answer. Those are the four things a change to the backend contract breaks
 * first, and nothing else in the client would notice.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { api, setUnauthorizedHandler, tokenStorage } from './api';

const BASE_URL = 'http://localhost:8001/api/v1';

const errorBody = (code: string) => JSON.stringify({ error: { error_code: code } });

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status });

/** A response whose body arrives in exactly these chunks, as the network delivers it. */
const streamResponse = (chunks: string[]) => {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)));
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
};

/** Minimal `localStorage`: the module keeps the token there, and Node has no DOM. */
const memoryStorage = () => {
  const entries = new Map<string, string>();
  return {
    getItem: (key: string) => entries.get(key) ?? null,
    setItem: (key: string, value: string) => void entries.set(key, value),
    removeItem: (key: string) => void entries.delete(key),
  };
};

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  vi.stubGlobal('localStorage', memoryStorage());
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  setUnauthorizedHandler(null);
  vi.unstubAllGlobals();
});

describe('authenticated requests', () => {
  it('sends the stored token as a bearer header', async () => {
    tokenStorage.set('token-123');
    fetchMock.mockResolvedValue(jsonResponse([]));

    await api.listDocuments();

    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE_URL}/documents/`);
    expect(options.headers.Authorization).toBe('Bearer token-123');
  });

  it('sends no authorization header when there is no token', async () => {
    fetchMock.mockResolvedValue(jsonResponse([]));

    await api.listDocuments();

    expect(fetchMock.mock.calls[0][1].headers.Authorization).toBeUndefined();
  });

  it('drops the token and notifies the app on 401', async () => {
    tokenStorage.set('expired');
    const onUnauthorized = vi.fn();
    setUnauthorizedHandler(onUnauthorized);
    fetchMock.mockResolvedValue(new Response('', { status: 401 }));

    await expect(api.listDocuments()).rejects.toThrow('UNAUTHORIZED');
    expect(tokenStorage.get()).toBeNull();
    expect(onUnauthorized).toHaveBeenCalledOnce();
  });
});

describe('error responses', () => {
  it('rejects a failed delete with the error code from the envelope', async () => {
    fetchMock.mockResolvedValue(new Response(errorBody('RATE_LIMIT_EXCEEDED'), { status: 429 }));

    await expect(api.deleteDocument('doc-1')).rejects.toThrow('RATE_LIMIT_EXCEEDED');
  });

  it('falls back to GENERIC when the error body is not the expected envelope', async () => {
    fetchMock.mockResolvedValue(new Response('<html>502 Bad Gateway</html>', { status: 502 }));

    await expect(api.listSummaries()).rejects.toThrow('GENERIC');
  });

  it('rejects a failed registration instead of returning an empty body', async () => {
    fetchMock.mockResolvedValue(new Response(errorBody('EMAIL_ALREADY_REGISTERED'), { status: 409 }));

    await expect(api.register('taken@example.com', 'secret123')).rejects.toThrow(
      'EMAIL_ALREADY_REGISTERED',
    );
  });

  it('percent-encodes the id it puts in the path', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));

    await api.deleteDocument('user::my notes.pdf');

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE_URL}/documents/user%3A%3Amy%20notes.pdf`,
    );
  });
});

describe('upload', () => {
  class FakeXhr {
    static last: FakeXhr;
    upload: { onprogress?: (event: ProgressEvent) => void } = {};
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    status = 0;
    responseText = '';
    headers: Record<string, string> = {};

    open(): void {}
    setRequestHeader(name: string, value: string): void {
      this.headers[name] = value;
    }
    send(): void {
      FakeXhr.last = this;
    }
  }

  const upload = () => api.uploadDocument(new File(['pdf bytes'], 'notes.pdf'));

  beforeEach(() => {
    vi.stubGlobal('XMLHttpRequest', FakeXhr);
  });

  it('resolves with the parsed body on success', async () => {
    const pending = upload();

    FakeXhr.last.status = 201;
    FakeXhr.last.responseText = JSON.stringify({ document_id: 'doc-1' });
    FakeXhr.last.onload?.();

    await expect(pending).resolves.toEqual({ document_id: 'doc-1' });
  });

  it('rejects with the error code from the envelope, not with `detail`', async () => {
    const pending = upload();

    FakeXhr.last.status = 400;
    FakeXhr.last.responseText = errorBody('UNSUPPORTED_FILE_TYPE');
    FakeXhr.last.onload?.();

    await expect(pending).rejects.toThrow('UNSUPPORTED_FILE_TYPE');
  });

  it('drops the token and notifies the app on 401', async () => {
    tokenStorage.set('expired');
    const onUnauthorized = vi.fn();
    setUnauthorizedHandler(onUnauthorized);
    const pending = upload();

    FakeXhr.last.status = 401;
    FakeXhr.last.onload?.();

    await expect(pending).rejects.toThrow('UNAUTHORIZED');
    expect(tokenStorage.get()).toBeNull();
    expect(onUnauthorized).toHaveBeenCalledOnce();
  });
});

describe('chat stream', () => {
  const collect = async (chunks: string[]) => {
    fetchMock.mockResolvedValue(streamResponse(chunks));
    const tokens: string[] = [];
    const done = vi.fn();

    await api.chatStream('question', [], undefined, (token) => tokens.push(token), done);

    return { tokens, done };
  };

  it('reads events split across chunk boundaries', async () => {
    const { tokens, done } = await collect([
      '{"type":"token","content":"Hel',
      'lo"}\n{"type":"token","content":" world"}\n',
      '{"type":"done","conversation_id":"conv-1","sources":["a.pdf"]}\n',
    ]);

    expect(tokens).toEqual(['Hello', ' world']);
    expect(done).toHaveBeenCalledWith({ conversation_id: 'conv-1', sources: ['a.pdf'] });
  });

  it('delivers the last event when the stream ends without a newline', async () => {
    const { done } = await collect([
      '{"type":"token","content":"x"}\n{"type":"done","conversation_id":"conv-2","sources":[]}',
    ]);

    expect(done).toHaveBeenCalledWith({ conversation_id: 'conv-2', sources: [] });
  });

  it('drops the token and notifies the app on 401', async () => {
    tokenStorage.set('expired');
    const onUnauthorized = vi.fn();
    setUnauthorizedHandler(onUnauthorized);
    fetchMock.mockResolvedValue(new Response('', { status: 401 }));

    await expect(
      api.chatStream('question', [], undefined, vi.fn(), vi.fn()),
    ).rejects.toThrow('UNAUTHORIZED');
    expect(tokenStorage.get()).toBeNull();
    expect(onUnauthorized).toHaveBeenCalledOnce();
  });

  it('rejects when the stream response is not ok', async () => {
    fetchMock.mockResolvedValue(new Response(errorBody('RATE_LIMIT_EXCEEDED'), { status: 429 }));

    await expect(api.chatStream('question', [], undefined, vi.fn(), vi.fn())).rejects.toThrow(
      'Chat stream failed with status 429',
    );
  });
});
