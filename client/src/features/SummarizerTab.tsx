import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import { api, DocumentInfo, SummaryInfo } from '../core/api';
import { useLanguage } from '../contexts/LanguageContext';
import Markdown from '../components/Markdown';
import { confirmDelete } from '../components/confirmDelete';
import { TrashIcon } from '../components/icons';

const SummarizerTab: React.FC = () => {
  const { t } = useLanguage();
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [summaries, setSummaries] = useState<SummaryInfo[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const fetchDocuments = async () => {
    try {
      const docs = await api.listDocuments();
      if (Array.isArray(docs)) {
        setDocuments(docs);
      } else {
        console.error('API did not return an array', docs);
      }
    } catch (error) {
      console.error('Error fetching docs', error);
    }
  };

  const fetchSummaries = async () => {
    try {
      const data = await api.listSummaries();
      if (Array.isArray(data)) {
        setSummaries(data);
      }
    } catch (error) {
      console.error('Error fetching summaries', error);
    }
  };

  useEffect(() => {
    fetchDocuments();
    fetchSummaries();
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    const file = e.target.files[0];
    setUploading(true);
    setUploadProgress(0);

    let processingInterval: ReturnType<typeof setInterval> | undefined = undefined;

    try {
      await api.uploadDocument(file, (percent) => {
        const p = Math.round(percent);
        const scaled = Math.min(Math.round(p / 2), 50);
        setUploadProgress(scaled);

        if (p >= 100 && !processingInterval) {
          processingInterval = setInterval(() => {
            setUploadProgress(prev => {
              if (prev >= 95) {
                clearInterval(processingInterval!);
                return 95;
              }
              return prev + 1;
            });
          }, 300);
        }
      });

      if (processingInterval) clearInterval(processingInterval);
      setUploadProgress(100);

      await fetchDocuments();
      toast.success(t('summarizer.upload_success'));
      e.target.value = '';
    } catch (error) {
      if (processingInterval) clearInterval(processingInterval);
      console.error(error);
      toast.error(t('summarizer.upload_error'));
    } finally {
      setTimeout(() => {
        setUploading(false);
        setUploadProgress(0);
      }, 1000);
    }
  };

  const handleToggleDoc = (id: string) => {
    setSelectedDocs(prev =>
      prev.includes(id) ? prev.filter(d => d !== id) : [...prev, id]
    );
  };

  const handleSelectAll = () => {
    if (selectedDocs.length === documents.length && documents.length > 0) {
      setSelectedDocs([]);
    } else {
      setSelectedDocs(documents.map(d => d.id));
    }
  };

  const handleDeleteDoc = (id: string, event: React.MouseEvent) => {
    event.stopPropagation();
    event.preventDefault();

    confirmDelete(t, async () => {
      try {
        await api.deleteDocument(id);
        toast.success(t('summarizer.delete_doc_success') || 'Dokument usunięty');
        await fetchDocuments();
        setSelectedDocs(prev => prev.filter(d => d !== id));
      } catch (error) {
        toast.error(t('summarizer.delete_doc_error') || 'Błąd podczas usuwania');
      }
    });
  };

  const handleDeleteSummary = (id: string, event: React.MouseEvent) => {
    event.stopPropagation();

    confirmDelete(t, async () => {
      try {
        await api.deleteSummary(id);
        toast.success(t('summarizer.delete_summary_success') || 'Podsumowanie usunięte');
        await fetchSummaries();
        const summaryToDelete = summaries.find(s => s.id === id);
        if (summaryToDelete && summary === summaryToDelete.summary) {
          setSummary('');
        }
      } catch (error) {
        toast.error(t('summarizer.delete_summary_error') || 'Błąd podczas usuwania');
      }
    });
  };

  const handleSummarize = async () => {
    if (selectedDocs.length === 0) {
      toast.error(t('summarizer.select_at_least_one'));
      return;
    }
    setLoading(true);
    try {
      const response = await api.summarize(selectedDocs);
      if (response && response.summary) {
        setSummary(response.summary);
        await fetchSummaries();
      } else {
        console.error('API did not return a summary', response);
        toast.error(t('summarizer.api_error'));
      }
    } catch (error) {
      console.error(error);
      toast.error(t('summarizer.error'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-5 max-w-6xl mx-auto">
      <h3 className="text-xl font-semibold mb-6 text-gray-800 dark:text-gray-200">{t('summarizer.title')}</h3>

      <div className="mb-8 p-6 border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-xl bg-gray-50 dark:bg-gray-800 transition-colors">
        <h4 className="text-sm font-bold uppercase tracking-wider text-gray-600 dark:text-gray-400 mb-3">{t('summarizer.upload_title')}</h4>
        <div className="flex items-center gap-4">
          <input
            type="file"
            onChange={handleFileUpload}
            disabled={uploading}
            className="block w-full text-sm text-gray-500 dark:text-gray-400
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100
              dark:file:bg-gray-700 dark:file:text-blue-300 transition-all"
          />
          {uploading && (
            <div className="flex-1 flex flex-col gap-1">
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs font-medium text-blue-600 animate-pulse">{t('summarizer.uploading')}</span>
                <span className="text-xs font-bold text-blue-600">{uploadProgress}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                <div
                  className="bg-blue-600 h-1.5 rounded-full transition-all duration-300 ease-out"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="flex flex-col">
          <div className="flex justify-between items-center mb-3">
            <h4 className="text-sm font-bold uppercase tracking-wider text-gray-600 dark:text-gray-400">{t('summarizer.kb_title')}</h4>
            {documents.length > 0 && (
              <div className="flex items-center gap-4">
                <span className="text-[10px] font-bold px-2 py-0.5 bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 rounded-full">
                  {selectedDocs.length}/{documents.length}
                </span>
                <label className="flex items-center gap-2 cursor-pointer group">
                  <div className="relative flex items-center">
                    <input
                      type="checkbox"
                      checked={selectedDocs.length === documents.length && documents.length > 0}
                      onChange={handleSelectAll}
                      className="peer h-4 w-4 cursor-pointer transition-all appearance-none rounded border border-slate-300 checked:bg-blue-600 checked:border-blue-600 hover:shadow-md focus:ring-2 focus:ring-blue-500/20"
                    />
                    <svg
                      className="absolute h-3.5 w-3.5 text-white opacity-0 peer-checked:opacity-100 pointer-events-none top-2/4 left-2/4 -translate-y-2/4 -translate-x-2/4"
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="4"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                  </div>
                  <span className="text-xs font-semibold text-gray-500 group-hover:text-blue-600 transition-colors">
                    {t('summarizer.select_all') || 'Zaznacz wszystko'}
                  </span>
                </label>
              </div>
            )}
          </div>
          <div className="flex-1 min-h-[200px] max-h-[300px] overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-lg p-3 bg-white dark:bg-gray-800 shadow-sm transition-colors">
            {Array.isArray(documents) && documents.length === 0 && (
              <div className="text-gray-500 italic text-center py-10">{t('summarizer.no_docs')}</div>
            )}
            {Array.isArray(documents) && documents.map(doc => (
              <div
                key={doc.id}
                className={`mb-1 last:mb-0 group flex items-center justify-between p-2 rounded-md transition-all border ${
                  selectedDocs.includes(doc.id)
                    ? 'bg-blue-50/50 dark:bg-blue-900/10 border-blue-200 dark:border-blue-800/50'
                    : 'hover:bg-gray-50 dark:hover:bg-gray-700/50 border-transparent'
                }`}
              >
                <label className="flex items-center gap-3 cursor-pointer flex-1 min-w-0">
                  <div className="relative flex items-center">
                    <input
                      type="checkbox"
                      checked={selectedDocs.includes(doc.id)}
                      onChange={() => handleToggleDoc(doc.id)}
                      className="peer h-5 w-5 cursor-pointer transition-all appearance-none rounded-md border border-slate-300 dark:border-gray-600 checked:bg-blue-600 checked:border-blue-600 hover:shadow-md focus:ring-2 focus:ring-blue-500/20"
                    />
                    <svg
                      className="absolute h-3.5 w-3.5 text-white opacity-0 peer-checked:opacity-100 pointer-events-none top-2/4 left-2/4 -translate-y-2/4 -translate-x-2/4"
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="4"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                  </div>
                  <div className="flex items-center gap-2 min-w-0">
                    <svg xmlns="http://www.w3.org/2000/svg" className={`h-4 w-4 flex-shrink-0 ${selectedDocs.includes(doc.id) ? 'text-blue-600' : 'text-gray-400'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <span className={`text-sm truncate transition-colors ${selectedDocs.includes(doc.id) ? 'text-blue-900 dark:text-blue-100 font-medium' : 'text-gray-700 dark:text-gray-300'}`}>
                      {doc.filename}
                    </span>
                  </div>
                </label>
                <button
                  onClick={(e) => handleDeleteDoc(doc.id, e)}
                  className="opacity-0 group-hover:opacity-100 p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-md transition-all"
                  title="Usuń dokument"
                >
                  <TrashIcon className="h-4 w-4" />
                </button>
              </div>
            ))}
            {!Array.isArray(documents) && <div className="text-red-500 text-center py-10">{t('summarizer.error_loading')}</div>}
          </div>
          <button
            onClick={handleSummarize}
            disabled={loading || selectedDocs.length === 0}
            className="mt-4 px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium rounded-lg transition-all shadow-sm flex items-center justify-center gap-2"
          >
            {loading && <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>}
            {t('summarizer.summarize_btn')}
          </button>
        </div>

        <div className="flex flex-col">
          <h4 className="text-sm font-bold uppercase tracking-wider text-gray-600 dark:text-gray-400 mb-3">{t('summarizer.result_title')}</h4>
          <div className="flex-1 min-h-[200px] border border-gray-200 dark:border-gray-700 rounded-lg p-6 bg-amber-50 dark:bg-gray-800 shadow-sm transition-colors">
            {loading && (
              <div className="flex flex-col items-center justify-center h-full gap-3 py-10">
                <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                <div className="text-sm text-blue-600 font-medium">{t('summarizer.generating')}</div>
              </div>
            )}
            {!loading && summary && (
              <div className="text-gray-800 dark:text-gray-200 leading-relaxed">
                <Markdown content={summary} />
              </div>
            )}
            {!loading && !summary && (
              <div className="text-gray-400 italic text-center py-20">{t('summarizer.select_prompt')}</div>
            )}
          </div>
        </div>
      </div>

      <div className="mt-12">
        <h4 className="text-sm font-bold uppercase tracking-wider text-gray-600 dark:text-gray-400 mb-4">{t('summarizer.history_title')}</h4>
        <div className="grid grid-cols-1 gap-4">
          {summaries.map(s => (
            <div
              key={s.id}
              className="group p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 shadow-sm cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors relative"
              onClick={() => {
                setSummary(s.summary);
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
            >
              <div className="flex justify-between items-start mb-2">
                <div className="text-xs font-medium text-blue-600 dark:text-blue-400">
                  {new Date(s.created_at).toLocaleString()}
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-xs text-gray-500 italic">
                    {s.document_ids.length} dokument(y)
                  </div>
                  <button
                    onClick={(e) => handleDeleteSummary(s.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 transition-all"
                    title="Usuń podsumowanie"
                  >
                    <TrashIcon className="h-4 w-4" />
                  </button>
                </div>
              </div>
              <div className="text-sm text-gray-700 dark:text-gray-300 line-clamp-2 pr-6">
                {s.summary}
              </div>
            </div>
          ))}
          {summaries.length === 0 && (
            <div className="text-gray-500 italic text-center py-4">{t('summarizer.no_history')}</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SummarizerTab;
