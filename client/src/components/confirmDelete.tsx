import toast from 'react-hot-toast';
import { WarningTriangleIcon } from './icons';

type TranslateFunction = (key: string) => string;

export const confirmDelete = (
  translate: TranslateFunction,
  onConfirm: () => void | Promise<void>,
): void => {
  toast(
    (activeToast) => (
      <div className="flex items-start gap-3 p-1">
        <div className="flex-shrink-0 mt-0.5">
          <WarningTriangleIcon className="h-6 w-6 text-amber-500" />
        </div>
        <div className="flex flex-col gap-3">
          <p className="text-sm font-semibold">
            {translate('common.confirm_delete') || 'Czy na pewno chcesz usunąć?'}
          </p>
          <div className="flex justify-end gap-2">
            <button
              onClick={() => toast.dismiss(activeToast.id)}
              className="px-3 py-1.5 text-xs font-semibold bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              {translate('common.cancel') || 'Anuluj'}
            </button>
            <button
              onClick={async () => {
                toast.dismiss(activeToast.id);
                await onConfirm();
              }}
              className="px-3 py-1.5 text-xs font-semibold bg-red-500 text-white rounded-md hover:bg-red-600 transition-colors shadow-md"
            >
              {translate('common.yes_delete') || 'Tak, usuń!'}
            </button>
          </div>
        </div>
      </div>
    ),
    { duration: 6000, position: 'top-center' },
  );
};
