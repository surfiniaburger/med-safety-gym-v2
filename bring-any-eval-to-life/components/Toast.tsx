import React, { createContext, useContext, useState, useCallback, ReactNode, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { XMarkIcon, ExclamationTriangleIcon, InformationCircleIcon, CheckCircleIcon } from '@heroicons/react/24/outline';

type ToastType = 'info' | 'success' | 'warning' | 'error';

interface Toast {
  id: string;
  message: string;
  type: ToastType;
  duration?: number;
}

interface ToastContextType {
  showToast: (message: string, type?: ToastType, duration?: number) => void;
  hideToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
};

export const ToastProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const hideToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const showToast = useCallback((message: string, type: ToastType = 'info', duration: number = 5000) => {
    const id = crypto.randomUUID();
    setToasts((prev) => [...prev, { id, message, type, duration }]);
  }, []);

  return (
    <ToastContext.Provider value={{ showToast, hideToast }}>
      {children}
      <div className="fixed bottom-6 right-6 z-[200] flex flex-col gap-3 pointer-events-none">
        <AnimatePresence mode="popLayout">
          {toasts.map((toast) => (
            <ToastItem key={toast.id} toast={toast} onHide={hideToast} />
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
};

const ToastItem: React.FC<{ toast: Toast; onHide: (id: string) => void }> = ({ toast, onHide }) => {
  useEffect(() => {
    if (toast.duration && toast.duration !== Infinity) {
      const timerId = setTimeout(() => {
        onHide(toast.id);
      }, toast.duration);

      return () => clearTimeout(timerId);
    }
  }, [toast.id, toast.duration, onHide]);

  const icons = {
    info: <InformationCircleIcon className="w-5 h-5 text-blue-400" />,
    success: <CheckCircleIcon className="w-5 h-5 text-emerald-400" />,
    warning: <ExclamationTriangleIcon className="w-5 h-5 text-amber-400" />,
    error: <ExclamationTriangleIcon className="w-5 h-5 text-rose-400" />,
  };

  const borderColors = {
    info: 'border-blue-500/20',
    success: 'border-emerald-500/20',
    warning: 'border-amber-500/20',
    error: 'border-rose-500/20',
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.9, filter: 'blur(10px)' }}
      animate={{ opacity: 1, y: 0, scale: 1, filter: 'blur(0px)' }}
      exit={{ opacity: 0, scale: 0.9, filter: 'blur(10px)', transition: { duration: 0.2 } }}
      className={`
        pointer-events-auto
        min-w-[320px] max-w-md
        bg-zinc-950/80 backdrop-blur-xl
        border ${borderColors[toast.type]}
        rounded-2xl p-4 shadow-2xl
        flex items-start gap-4
        relative overflow-hidden
      `}
    >
      {/* Progress bar for auto-hide */}
      {toast.duration !== Infinity && (
        <motion.div
          initial={{ width: '100%' }}
          animate={{ width: '0%' }}
          transition={{ duration: (toast.duration || 5000) / 1000, ease: 'linear' }}
          className={`absolute bottom-0 left-0 h-0.5 bg-current opacity-20`}
        />
      )}

      <div className="flex-shrink-0 mt-0.5">
        {icons[toast.type]}
      </div>

      <div className="flex-grow">
        <p className="text-sm font-medium text-zinc-200 leading-relaxed">
          {toast.message}
        </p>
      </div>

      <button
        onClick={() => onHide(toast.id)}
        className="flex-shrink-0 text-zinc-500 hover:text-zinc-300 transition-colors p-1 -mr-1 -mt-1"
      >
        <XMarkIcon className="w-5 h-5" />
      </button>
    </motion.div>
  );
};
