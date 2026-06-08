import React, { useState, useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import ChatTab from './features/ChatTab';
import QATab from './features/QATab';
import SummarizerTab from './features/SummarizerTab';
import FAQTab from './features/FAQTab';
import LoginScreen from './features/LoginScreen';
import { LanguageProvider, useLanguage } from './contexts/LanguageContext';
import { AuthProvider, useAuth } from './contexts/AuthContext';

const AppContent: React.FC = () => {
  const { language, setLanguage, t } = useLanguage();
  const { isAuthenticated, email, logout } = useAuth();
  const [activeTab, setActiveTab] = useState<'chat' | 'qa' | 'summarizer' | 'faq'>('chat');
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode(!darkMode);

  const tabClass = (id: string) => `px-5 py-2.5 cursor-pointer transition-colors duration-200 border-b-2 ${
    activeTab === id 
      ? 'border-blue-500 text-blue-600 dark:text-blue-400 font-bold' 
      : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
  }`;

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 font-sans transition-colors duration-300">
      <header className="bg-slate-800 dark:bg-black p-4 flex justify-between items-center shadow-md">
        <h1 className="text-xl font-bold text-white">{t('app.title')}</h1>
        <div className="flex items-center gap-4">
          <div className="flex bg-gray-200 dark:bg-gray-700 rounded-lg p-1">
            <button
              onClick={() => setLanguage('pl')}
              className={`px-3 py-1 rounded-md text-xs font-bold transition-colors ${
                language === 'pl' ? 'bg-white dark:bg-gray-500 shadow-sm' : 'text-gray-500 dark:text-gray-400'
              }`}
            >
              PL
            </button>
            <button
              onClick={() => setLanguage('en')}
              className={`px-3 py-1 rounded-md text-xs font-bold transition-colors ${
                language === 'en' ? 'bg-white dark:bg-gray-500 shadow-sm' : 'text-gray-500 dark:text-gray-400'
              }`}
            >
              EN
            </button>
          </div>
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-lg bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            aria-label="Toggle Dark Mode"
          >
            {darkMode ? '🌙' : '☀️'}
          </button>
          {isAuthenticated && (
            <div className="flex items-center gap-3">
              <span className="text-sm text-gray-300 hidden sm:inline" title={email ?? undefined}>
                {email}
              </span>
              <button
                onClick={logout}
                className="px-3 py-1.5 rounded-lg bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-sm font-medium transition-colors"
              >
                {t('auth.logout')}
              </button>
            </div>
          )}
        </div>
      </header>

      {isAuthenticated ? (
        <>
          <nav className="flex bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 sticky top-0 z-10">
            <div className={tabClass('chat')} onClick={() => setActiveTab('chat')}>{t('nav.chat')}</div>
            <div className={tabClass('qa')} onClick={() => setActiveTab('qa')}>{t('nav.qa')}</div>
            <div className={tabClass('summarizer')} onClick={() => setActiveTab('summarizer')}>{t('nav.summarizer')}</div>
            <div className={tabClass('faq')} onClick={() => setActiveTab('faq')}>{t('nav.faq')}</div>
          </nav>

          <main className="p-4">
            {activeTab === 'chat' && <ChatTab />}
            {activeTab === 'qa' && <QATab />}
            {activeTab === 'summarizer' && <SummarizerTab />}
            {activeTab === 'faq' && <FAQTab />}
          </main>
        </>
      ) : (
        <main className="p-4">
          <LoginScreen />
        </main>
      )}
      <Toaster
        position="bottom-right"
        toastOptions={{
          style: darkMode ? {
            background: '#1f2937',
            color: '#f3f4f6',
            border: '1px solid #374151',
          } : {
            background: '#ffffff',
            color: '#111827',
            border: '1px solid #e5e7eb',
          },
        }}
      />
    </div>
  );
};

const App: React.FC = () => (
  <LanguageProvider>
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  </LanguageProvider>
);

export default App;
