import React, { useState } from 'react';
import ChatTab from './ChatTab';
import QATab from './QATab';
import SummarizerTab from './SummarizerTab';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'qa' | 'summarizer'>('chat');

  const tabStyle = (id: string) => ({
    padding: '10px 20px',
    cursor: 'pointer',
    borderBottom: activeTab === id ? '2px solid blue' : 'none',
    fontWeight: activeTab === id ? 'bold' : 'normal',
  });

  return (
    <div style={{ fontFamily: 'Arial, sans-serif' }}>
      <header style={{ background: '#282c34', padding: '10px 20px', color: 'white' }}>
        <h1>Docs Analysis Tool</h1>
      </header>

      <nav style={{ display: 'flex', background: '#f8f9fa', borderBottom: '1px solid #dee2e6' }}>
        <div style={tabStyle('chat')} onClick={() => setActiveTab('chat')}>Chat</div>
        <div style={tabStyle('qa')} onClick={() => setActiveTab('qa')}>Q&A</div>
        <div style={tabStyle('summarizer')} onClick={() => setActiveTab('summarizer')}>Summarizer</div>
      </nav>

      <main>
        {activeTab === 'chat' && <ChatTab />}
        {activeTab === 'qa' && <QATab />}
        {activeTab === 'summarizer' && <SummarizerTab />}
      </main>
    </div>
  );
};

export default App;
