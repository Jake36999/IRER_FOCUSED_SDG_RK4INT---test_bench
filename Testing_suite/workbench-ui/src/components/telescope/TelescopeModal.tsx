import React, { useEffect } from 'react';
import { createPortal } from 'react-dom';

interface TelescopeModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  triggerEventId?: string;
  // Function as child: receives close handler
  children: (args: { close: () => void }) => React.ReactNode;
}

export const TelescopeModal: React.FC<TelescopeModalProps> = ({
  isOpen,
  onClose,
  title,
  triggerEventId,
  children
}) => {
  // Global Escape Key Listener
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm">
      {/* Modal Shell */}
      <div 
        className="bg-slate-900 border border-slate-700 w-[90vw] h-[85vh] shadow-2xl flex flex-col rounded-lg overflow-hidden animate-in fade-in zoom-in-95 duration-200"
        role="dialog"
        aria-modal="true"
      >
        {/* Header - Draggable Area */}
        <div className="h-14 bg-slate-950 border-b border-slate-800 flex items-center justify-between px-6 select-none">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/20 rounded-full">
              {/* Telescope Icon Placeholder */}
              <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" /><path d="M12 8v4l3 3" /></svg>
            </div>
            <div>
              <span className="text-slate-100 font-mono text-lg">{title}</span>
              <div className="text-slate-500 text-xs font-mono">
                Observation ID: {triggerEventId || "LIVE_STREAM"}
              </div>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="text-slate-500 hover:text-red-400 transition-colors font-mono text-xs"
          >
            [ESC] CLOSE
          </button>
        </div>

        {/* Content Area - Dynamic Injection via Render Prop */}
        <div className="flex-1 overflow-auto bg-[#050505] relative p-6 custom-scrollbar">
          {/* Grid Background Effect */}
          <div className="absolute inset-0 pointer-events-none opacity-20" 
               style={{ backgroundImage: 'radial-gradient(#333 1px, transparent 1px)', backgroundSize: '20px 20px' }} 
          />
          <div className="relative z-10">
            {children({ close: onClose })}
          </div>
        </div>

        {/* Footer - Context Metadata */}
        <div className="h-10 bg-slate-950 border-t border-slate-800 flex items-center px-6 text-xs text-slate-500 font-mono justify-between">
          <span>MODE: DEEP_INSPECTION</span>
          <span>PRESS [ESC] TO RETURN TO ORBIT</span>
        </div>
      </div>
    </div>,
    document.body
  );
};