import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import { api, ChatMessage, ConversationInfo } from '../core/api';
import { useLanguage } from '../contexts/LanguageContext';
import Markdown from '../components/Markdown';
import { confirmDelete } from '../components/confirmDelete';
import { TrashIcon, AssistantRobotIcon } from '../components/icons';

const AssistantAvatar = () => (
  <div className="flex-shrink-0 mr-2 mt-0.5">
    <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center border border-blue-200 dark:border-blue-800 shadow-sm">
      <AssistantRobotIcon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
    </div>
  </div>
);

const ChatTab: React.FC = () => {
  const { t } = useLanguage();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversations, setConversations] = useState<ConversationInfo[]>([]);
  const [currentConvId, setCurrentConvId] = useState<string | undefined>(undefined);

  const fetchConversations = async () => {
    try {
      const data = await api.listConversations();
      if (Array.isArray(data)) {
        setConversations(data);
      }
    } catch (error) {
      console.error('Error fetching conversations:', error);
    }
  };

  useEffect(() => {
    fetchConversations();
  }, []);

  const handleSend = async () => {
    if (!input.trim()) return;

    const currentInput = input;
    const userMsg: ChatMessage = { role: 'user', content: currentInput };

    setMessages(prev => [...prev, userMsg, { role: 'assistant', content: '' }]);
    setInput('');
    setLoading(true);

    const appendToAssistant = (token: string) => {
      setMessages(prev => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last && last.role === 'assistant') {
          next[next.length - 1] = { ...last, content: last.content + token };
        }
        return next;
      });
    };

    try {
      await api.chatStream(
        currentInput,
        currentConvId,
        (token) => {
          setLoading(false);
          appendToAssistant(token);
        },
        (data) => {
          if (data.sources && data.sources.length === 0) {
            toast.error(t('chat.no_docs'), { duration: 4000 });
          }
          if (!currentConvId) {
            setCurrentConvId(data.conversation_id);
            fetchConversations();
          }
        },
      );
    } catch (error) {
      console.error(error);
      toast.error(t('chat.error'));
      setMessages(prev => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last && last.role === 'assistant' && last.content === '') {
          next.pop();
        }
        return next;
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSelectConversation = async (id: string) => {
    setLoading(true);
    try {
      const conv = await api.getConversation(id);
      setMessages(conv.messages);
      setCurrentConvId(conv.id);
    } catch (error) {
      toast.error(t('chat.load_error'));
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteConversation = (id: string, event: React.MouseEvent) => {
    event.stopPropagation();

    confirmDelete(t, async () => {
      try {
        await api.deleteConversation(id);
        toast.success(t('chat.delete_success'));
        fetchConversations();
        if (currentConvId === id) {
          handleNewChat();
        }
      } catch (error) {
        toast.error(t('chat.delete_error'));
      }
    });
  };

  const handleNewChat = () => {
    setMessages([]);
    setCurrentConvId(undefined);
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      toast.success(t('chat.copy_success'));
    }).catch(err => {
      console.error('Failed to copy: ', err);
    });
  };

  const handleExportMarkdown = (content: string) => {
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat_message_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '_')}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-[calc(100vh-150px)] max-w-full gap-8 p-5 px-10">
      <div className="w-64 flex flex-col border-r border-gray-200 dark:border-gray-700 pr-4">
        <button
          onClick={handleNewChat}
          className="mb-4 p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
        >
          + {t('chat.new_chat')}
        </button>
        <div className="flex-1 overflow-y-auto">
          <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-2">
            {t('chat.history')}
          </h4>
          {conversations.map(conv => (
            <div
              key={conv.id}
              onClick={() => handleSelectConversation(conv.id)}
              className={`group flex items-center justify-between p-2 mb-1 rounded cursor-pointer transition-colors ${
                currentConvId === conv.id
                  ? 'bg-blue-50 dark:bg-gray-700 text-blue-700 dark:text-blue-300'
                  : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-400'
              }`}
            >
              <span className="text-sm truncate flex-1">{conv.title}</span>
              <button
                onClick={(e) => handleDeleteConversation(conv.id, e)}
                className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 transition-all"
              >
                <TrashIcon className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className="flex-1 flex flex-col max-w-5xl mx-auto w-full">
        <h3 className="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200">{t('chat.title')}</h3>
        <div className="flex-1 border border-gray-300 dark:border-gray-700 rounded-lg overflow-y-auto mb-4 p-4 bg-gray-50 dark:bg-gray-800 shadow-inner">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full text-gray-400 italic">
              {t('chat.start_prompt')}
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`flex mb-3 ${m.role === 'user' ? 'justify-end' : 'justify-start'} items-start`}>
              {m.role === 'assistant' && <AssistantAvatar />}
              <div className={`max-w-[80%] px-4 py-2 rounded-2xl text-sm relative group/msg ${
                m.role === 'user'
                  ? 'bg-blue-600 text-white rounded-tr-none'
                  : 'bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 border border-gray-200 dark:border-gray-600 rounded-tl-none shadow-sm'
              }`}>
                {m.role === 'user' ? (
                  <div className="whitespace-pre-wrap">{m.content}</div>
                ) : (
                  <Markdown content={m.content} />
                )}
                {m.role === 'assistant' && m.content && (
                  <div className="flex justify-end gap-2 mt-2 pt-2 border-t border-gray-100 dark:border-gray-600 opacity-0 group-hover/msg:opacity-100 transition-opacity">
                    <button
                      onClick={() => handleCopy(m.content)}
                      className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded transition-colors text-gray-500 dark:text-gray-400"
                      title={t('chat.copy')}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                      </svg>
                    </button>
                    <button
                      onClick={() => handleExportMarkdown(m.content)}
                      className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded transition-colors text-gray-500 dark:text-gray-400"
                      title={t('chat.export_md')}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start mb-3 items-start">
              <AssistantAvatar />
              <div className="bg-white dark:bg-gray-700 px-4 py-2 rounded-2xl rounded-tl-none border border-gray-200 dark:border-gray-600 shadow-sm">
                <div className="flex gap-1">
                  <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                  <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:0.4s]"></div>
                </div>
              </div>
            </div>
          )}
        </div>
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            className="flex-1 p-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
            placeholder={t('chat.placeholder')}
          />
          <button
            onClick={handleSend}
            disabled={loading}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium rounded-lg transition-colors shadow-sm"
          >
            {t('chat.send')}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatTab;
