import React, { useState } from 'react';
import toast from 'react-hot-toast';
import { api } from '../core/api';
import { useLanguage } from '../contexts/LanguageContext';
import Markdown from '../components/Markdown';

const QATab: React.FC = () => {
  const { t } = useLanguage();
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    if (!question.trim()) return;
    setLoading(true);
    try {
      const response = await api.askQuestion(question);

      if (response.sources && response.sources.length === 0) {
        toast.error(t('qa.no_docs'), {
          duration: 4000,
        });
      }

      setAnswer(response.answer);
    } catch (error) {
      console.error(error);
      toast.error(t('qa.error'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-5 max-w-4xl mx-auto">
      <h3 className="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200">{t('qa.title')}</h3>
      <div className="flex gap-2 mb-6">
        <input
          type="text"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleAsk()}
          className="flex-1 p-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
          placeholder={t('qa.placeholder')}
        />
        <button
          onClick={handleAsk}
          disabled={loading}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium rounded-lg transition-colors shadow-sm"
        >
          {t('qa.ask')}
        </button>
      </div>

      {loading && <div className="text-center text-gray-500 italic animate-pulse py-4">{t('qa.searching')}</div>}

      {answer && (
        <div className="border border-gray-200 dark:border-gray-700 p-6 rounded-xl bg-gray-50 dark:bg-gray-800 shadow-sm transition-colors">
          <strong className="block text-sm font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400 mb-2">{t('qa.answer')}</strong>
          <div className="text-gray-800 dark:text-gray-200 leading-relaxed">
            <Markdown content={answer} />
          </div>
        </div>
      )}
    </div>
  );
};

export default QATab;
