import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import toast from 'react-hot-toast';
import { api, tokenStorage, setUnauthorizedHandler } from '../core/api';
import { useLanguage } from './LanguageContext';

interface AuthContextType {
  email: string | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const EMAIL_KEY = 'authEmail';

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const { t } = useLanguage();
  const [token, setToken] = useState<string | null>(() => tokenStorage.get());
  const [email, setEmail] = useState<string | null>(() => localStorage.getItem(EMAIL_KEY));

  const logout = useCallback(() => {
    tokenStorage.clear();
    localStorage.removeItem(EMAIL_KEY);
    setToken(null);
    setEmail(null);
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(() => {
      localStorage.removeItem(EMAIL_KEY);
      setToken(null);
      setEmail(null);
      toast.error(t('auth.session_expired'));
    });
    return () => setUnauthorizedHandler(null);
  }, [t]);

  const applySession = (accessToken: string, userEmail: string) => {
    tokenStorage.set(accessToken);
    localStorage.setItem(EMAIL_KEY, userEmail);
    setToken(accessToken);
    setEmail(userEmail);
  };

  const login = async (emailInput: string, password: string) => {
    const result = await api.login(emailInput, password);
    applySession(result.access_token, result.email);
  };

  const register = async (emailInput: string, password: string) => {
    const result = await api.register(emailInput, password);
    applySession(result.access_token, result.email);
  };

  return (
    <AuthContext
      value={{ email, isAuthenticated: token !== null, login, register, logout }}
    >
      {children}
    </AuthContext>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
