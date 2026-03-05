import React, { ComponentType, useState, useEffect } from 'react';

// Props injected by the HOC
export interface FocusModeProps {
  focusModeActive: boolean;
  triggerFocus: () => void;
  clearFocus: () => void;
}

export function withFocusMode<P extends object>(
  WrappedComponent: ComponentType<P & FocusModeProps>
) {
  return (props: P) => {
    const [isFocusMode, setIsFocusMode] = useState(false);

    // Auto-trigger focus if global criticality (e.g. from context) is high
    // Mocking a global state subscription here
    useEffect(() => {
        // Example: window.addEventListener('CRITICAL_EVENT', () => setIsFocusMode(true));
    }, []);

    const triggerFocus = () => setIsFocusMode(true);
    const clearFocus = () => setIsFocusMode(false);

    return (
      <div className={`transition-all duration-700 ${isFocusMode ? 'bg-black' : ''}`}>
        {/* Dimming Overlay for Focus Mode */}
        {isFocusMode && (
           <div className="fixed inset-0 bg-black/80 z-[40] pointer-events-none animate-in fade-in duration-1000" />
        )}

        {/* The Wrapped Component gets the control props */}
        <div className={`relative ${isFocusMode ? 'z-[50] scale-105 transition-transform' : ''}`}>
           <WrappedComponent 
             {...props} 
             focusModeActive={isFocusMode}
             triggerFocus={triggerFocus}
             clearFocus={clearFocus}
           />
        </div>
        
        {/* Exit Button only visible in Focus Mode */}
        {isFocusMode && (
          <button 
            onClick={clearFocus}
            className="fixed top-8 left-1/2 -translate-x-1/2 z-[60] px-6 py-2 bg-slate-800 text-slate-200 border border-slate-600 rounded-full font-mono text-xs hover:bg-slate-700 transition-colors"
          >
            EXIT FOCUS MODE [ESC]
          </button>
        )}
      </div>
    );
  };
}