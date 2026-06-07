import React, { useState } from 'react';
import { api } from './api';

const QATab: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    if (!question.trim()) return;
    setLoading(true);
    try {
      const response = await api.askQuestion(question);
      setAnswer(response.answer);
    } catch (error) {
      console.error(error);
      alert('Error asking question');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h3>Q&A (One-shot)</h3>
      <div style={{ marginBottom: '10px' }}>
        <input
          type="text"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          style={{ width: '80%', padding: '10px' }}
          placeholder="Ask a question..."
        />
        <button onClick={handleAsk} disabled={loading} style={{ padding: '10px 20px' }}>
          Ask
        </button>
      </div>
      {loading && <div>Searching for answer...</div>}
      {answer && (
        <div style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '5px', background: '#f9f9f9' }}>
          <strong>Answer:</strong>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
};

export default QATab;
