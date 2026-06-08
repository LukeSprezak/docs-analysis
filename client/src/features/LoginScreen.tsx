import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useLanguage } from '../contexts/LanguageContext';

type Mode = 'login' | 'register';

const errorCodeToTranslationKey = (code: string): string => {
  switch (code) {
    case 'AUTHENTICATION_ERROR':
      return 'auth.error_invalid_credentials';
    case 'VALIDATION_ERROR':
      return 'auth.error_email_taken';
    default:
      return 'auth.error_generic';
  }
};

const LoginScreen: React.FC = () => {
  const { t } = useLanguage();
  const { login, register } = useAuth();
  const [mode, setMode] = useState<Mode>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [errorKey, setErrorKey] = useState<string | null>(null);

  const isLogin = mode === 'login';

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setErrorKey(null);

    if (!email.trim() || !password) {
      setErrorKey('auth.error_fields_required');
      return;
    }

    setSubmitting(true);
    try {
      if (isLogin) {
        await login(email.trim(), password);
      } else {
        await register(email.trim(), password);
      }
    } catch (error) {
      const code = error instanceof Error ? error.message : 'GENERIC';
      setErrorKey(errorCodeToTranslationKey(code));
    } finally {
      setSubmitting(false);
    }
  };

  const toggleMode = () => {
    setMode(isLogin ? 'register' : 'login');
    setErrorKey(null);
  };

  return (
    <div className="flex items-center justify-center min-h-[70vh] px-4">
      <div className="w-full max-w-md bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl shadow-lg p-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-1">
          {isLogin ? t('auth.login_title') : t('auth.register_title')}
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">{t('auth.subtitle')}</p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label
              htmlFor="auth-email"
              className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
            >
              {t('auth.email')}
            </label>
            <input
              id="auth-email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder={t('auth.email_placeholder')}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label
              htmlFor="auth-password"
              className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
            >
              {t('auth.password')}
            </label>
            <input
              id="auth-password"
              type="password"
              autoComplete={isLogin ? 'current-password' : 'new-password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={t('auth.password_placeholder')}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {errorKey && (
            <p className="text-sm text-red-600 dark:text-red-400" role="alert">
              {t(errorKey)}
            </p>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-semibold transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {submitting
              ? t('auth.submitting')
              : isLogin
                ? t('auth.login_btn')
                : t('auth.register_btn')}
          </button>
        </form>

        <button
          type="button"
          onClick={toggleMode}
          className="mt-5 w-full text-sm text-blue-600 dark:text-blue-400 hover:underline"
        >
          {isLogin ? t('auth.switch_to_register') : t('auth.switch_to_login')}
        </button>
      </div>
    </div>
  );
};

export default LoginScreen;
