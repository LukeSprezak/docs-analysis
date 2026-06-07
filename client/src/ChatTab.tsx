import React, { useState } from 'react';
import { api, ChatMessage } from './api';

const ChatTab: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg: ChatMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await api.chat(input, messages);
      const assistantMsg: ChatMessage = { role: 'assistant', content: response.answer };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (error) {
      console.error(error);
      alert('Error sending message');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h3>Chat (RAG)</h3>
      <div style={{ border: '1px solid #ccc', height: '400px', overflowY: 'scroll', marginBottom: '10px', padding: '10px' }}>
        {messages.map((m, i) => (
          <div key={i} style={{ textAlign: m.role === 'user' ? 'right' : 'left', margin: '5px 0' }}>
            <span style={{ background: m.role === 'user' ? '#e3f2fd' : '#f5f5f5', padding: '5px 10px', borderRadius: '10px' }}>
              {m.content}
            </span>
          </div>
        ))}
        {loading && <div>Thinking...</div>}
      </div>
      <div style={{ display: 'flex' }}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyPress={e => e.key === 'Enter' && handleSend()}
          style={{ flex: 1, padding: '10px' }}
          placeholder="Type your message..."
        />
        <button onClick={handleSend} disabled={loading} style={{ padding: '10px 20px' }}>
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatTab;
