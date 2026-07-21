import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { api } from '../core/api';

type Language = 'pl' | 'en';

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
  loading: boolean;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const LanguageProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [language, setLanguage] = useState<Language>(() => {
    const saved = localStorage.getItem('language');
    return (saved as Language) || 'pl';
  });

  const [translations, setTranslations] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTranslations = async () => {
      setLoading(true);
      try {
        const data = await api.getTranslations(language);
        setTranslations(data);
        localStorage.setItem('language', language);
      } catch (error) {
        console.error('Failed to fetch translations', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTranslations();
  }, [language]);

  const t = (key: string): string => {
    return translations[key] || key;
  };

  return (
    <LanguageContext value={{ language, setLanguage, t, loading }}>
      {loading && Object.keys(translations).length === 0 ? (
        <div className="min-h-screen flex items-center justify-center bg-white dark:bg-gray-900">
           <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
        </div>
      ) : children}
    </LanguageContext>
  );
};

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};
