import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';

const FAQTab: React.FC = () => {
  const { t } = useLanguage();

  return (
    <div className="p-5 max-w-4xl mx-auto text-gray-800 dark:text-gray-200">
      <h3 className="text-2xl font-bold mb-6 border-b pb-2 border-gray-200 dark:border-gray-700">{t('faq.title')}</h3>

      <div className="space-y-6">
        <section>
          <h4 className="text-lg font-semibold mb-2 text-blue-600 dark:text-blue-400">{t('faq.q1')}</h4>
          <ul className="list-disc ml-6 space-y-2">
            <li>{t('faq.a1_1')}</li>
            <li>{t('faq.a1_2')}</li>
            <li>{t('faq.a1_3')}</li>
          </ul>
        </section>

        <section>
          <h4 className="text-lg font-semibold mb-2 text-blue-600 dark:text-blue-400">{t('faq.q2')}</h4>
          <ul className="list-disc ml-6 space-y-2">
            <li>{t('faq.a2_1')}</li>
            <li>{t('faq.a2_2')}</li>
            <li>{t('faq.a2_3')}</li>
          </ul>
        </section>

        <section>
          <h4 className="text-lg font-semibold mb-2 text-blue-600 dark:text-blue-400">{t('faq.q3')}</h4>
          <ul className="list-disc ml-6 space-y-2">
            <li>{t('faq.a3_1')}</li>
            <li>{t('faq.a3_2')}</li>
          </ul>
        </section>

        <section>
          <h4 className="text-lg font-semibold mb-2 text-blue-600 dark:text-blue-400">{t('faq.q4')}</h4>
          <ul className="list-disc ml-6 space-y-2">
            <li>{t('faq.a4_1')}</li>
            <li>{t('faq.a4_2')}</li>
            <li>{t('faq.a4_3')}</li>
          </ul>
        </section>

        <section>
          <h4 className="text-lg font-semibold mb-2 text-blue-600 dark:text-blue-400">{t('faq.q5')}</h4>
          <ul className="list-disc ml-6 space-y-2">
            <li>{t('faq.a5_1')}</li>
            <li>{t('faq.a5_2')}</li>
          </ul>
        </section>
      </div>
    </div>
  );
};

export default FAQTab;
