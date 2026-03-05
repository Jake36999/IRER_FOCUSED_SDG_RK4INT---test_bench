import React, { createContext, useContext, useState, ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';

// --- Toast Definitions ---
type ToastType = 'info' | 'success' | 'warning' | 'critical';

interface Toast {
  id: string;
  title: string;
  message: string;
  type: ToastType;
}

interface SignalContextType {
  addToast: (title: string, message: string, type?: ToastType) => void;
}

const SignalContext = createContext<SignalContextType | undefined>(undefined);

export const useSignal = () => {
  const context = useContext(SignalContext);
  if (!context) throw new Error('useSignal must be used within a SignalLayer');
  return context;
};

// --- Presentational Component: The Toast Card ---
const ToastCard: React.FC<{ toast: Toast; onClose: (id: string) => void }> = ({ toast, onClose }) => {
  const borderColors = {
    info: 'border-blue-800',
    success: 'border-green-800',
    warning: 'border-amber-600',
    critical: 'border-red-600 animate-pulse',
  };
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.3 }}
      className={`mb-2 w-80 p-4 rounded-lg bg-[#0a0a0a]/95 backdrop-blur border-l-4 shadow-2xl transform transition-all duration-300 ${borderColors[toast.type]} border-y border-r border-slate-800`}
    >
      <div className="flex justify-between items-start">
        <h4 className={`font-bold text-sm ${toast.type === 'critical' ? 'text-red-400' : 'text-slate-200'}`}>{toast.title}</h4>
        <button onClick={() => onClose(toast.id)} className="text-slate-500 hover:text-white text-xs">✕</button>
      </div>
      <p className="text-xs text-slate-400 mt-1 font-mono">{toast.message}</p>
    </motion.div>
  );
};

// --- Consolidated Overlay Layer ---
interface OverlayProps {
  isCritical?: boolean;
  activePrimitives?: string[];
  children?: ReactNode;
}

export const SignalLayer: React.FC<OverlayProps> & { useSignal: typeof useSignal } = ({
  isCritical = false,
  activePrimitives = [],
  children
}) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = (title: string, message: string, type: ToastType = 'info') => {
    const id = Math.random().toString(36).substr(2, 9);
    setToasts((prev) => [...prev, { id, title, message, type }]);
    if (type !== 'critical') {
      setTimeout(() => removeToast(id), 5000);
    }
  };
  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  return (
    <SignalContext.Provider value={{ addToast }}>
      {children}
      {typeof document !== 'undefined' && createPortal(
        <div className="absolute inset-0 pointer-events-none z-50 overflow-hidden">
          {/* 1. Ghost Tracer: Visual Border Alarms */}
          <div className={`absolute inset-0 border-[4px] transition-colors duration-500 ${
            isCritical
              ? 'border-red-600/50 animate-pulse'
              : activePrimitives.length > 0
                ? 'border-fuchsia-600/30'
                : 'border-transparent'
          }`} />

          {/* 2. Primitive Execution Indicators (Top Right) */}
          <div className="absolute top-4 right-4 flex flex-col space-y-2 items-end">
            <AnimatePresence>
              {activePrimitives.map((prim) => (
                <motion.div
                  key={prim}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="px-3 py-1 bg-fuchsia-900/80 border border-fuchsia-500 text-fuchsia-100 font-mono text-xs rounded shadow-lg backdrop-blur"
                >
                  EXEC :: {prim}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          {/* 3. Toast Stack (Bottom Right) */}
          <div className="absolute bottom-4 right-4 flex flex-col space-y-2 items-end">
            <AnimatePresence>
              {toasts.map((toast) => (
                <ToastCard key={toast.id} toast={toast} onClose={removeToast} />
              ))}
            </AnimatePresence>
          </div>
        </div>,
        document.body
      )}
    </SignalContext.Provider>
  );
};

SignalLayer.useSignal = useSignal;