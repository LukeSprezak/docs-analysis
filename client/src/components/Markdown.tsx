import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const Markdown: React.FC<{ content: string }> = ({ content }) => (
  <ReactMarkdown
    remarkPlugins={[remarkGfm]}
    components={{
      p: ({ children }) => <p className="mb-1 last:mb-0">{children}</p>,
      ul: ({ children }) => <ul className="list-disc pl-5 mb-1 space-y-1">{children}</ul>,
      ol: ({ children }) => <ol className="list-decimal pl-5 mb-1 space-y-1">{children}</ol>,
      li: ({ children }) => <li>{children}</li>,
      strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
      em: ({ children }) => <em className="italic">{children}</em>,
      h1: ({ children }) => <h1 className="text-base font-bold mb-1 mt-2">{children}</h1>,
      h2: ({ children }) => <h2 className="text-sm font-bold mb-1 mt-2">{children}</h2>,
      h3: ({ children }) => <h3 className="text-sm font-semibold mb-1 mt-2">{children}</h3>,
      code: ({ children }) => (
        <code className="px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-[0.85em] font-mono">
          {children}
        </code>
      ),
      a: ({ href, children }) => (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 dark:text-blue-400 underline"
        >
          {children}
        </a>
      ),
      table: ({ children }) => (
        <div className="overflow-x-auto my-2">
          <table className="border-collapse text-xs">{children}</table>
        </div>
      ),
      th: ({ children }) => (
        <th className="border border-gray-300 dark:border-gray-600 px-2 py-1 font-semibold">
          {children}
        </th>
      ),
      td: ({ children }) => (
        <td className="border border-gray-300 dark:border-gray-600 px-2 py-1">{children}</td>
      ),
    }}
  >
    {content}
  </ReactMarkdown>
);

export default Markdown;
