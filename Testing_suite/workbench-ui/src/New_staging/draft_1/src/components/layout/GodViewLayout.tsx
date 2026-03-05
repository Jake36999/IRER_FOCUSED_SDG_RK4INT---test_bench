import React, { useState, useEffect } from 'react';
import { Sidebar } from '../shared/Sidebar';
import { DiagnosticContextBar } from '../hud/DiagnosticContextBar'; // Assumed split
import { MonitorContainer } from '../monitor/MonitorContainer';
import { RemedialManifold } from '../kel/RemedialManifold';
import { CommandPalette } from '../shared/CommandPalette';
import { SignalLayer } from '../monitor/SignalLayer';

// Lightswind-inspired layout shell
export const GodViewLayout: React.FC = () => {
  const [isManifoldOpen, setManifoldOpen] = useState(true);

  // Global Hotkey Listener for "Emergency Exit"
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        // Logic to kill deep-dive modals
        console.log('AXIOM 2: Emergency Exit Triggered');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="h-screen w-screen bg-[#0a0a0a] text-slate-200 overflow-hidden font-mono selection:bg-cyan-900 selection:text-cyan-50">
      {/* 1. Global Command Router (Ctrl+K) */}
      <CommandPalette />

      {/* 2. Unified Signal Overlay (Toasts, Ghost Tracer) */}
      <SignalLayer />

      <div className="grid h-full grid-cols-[240px_1fr_auto] grid-rows-[48px_1fr]">
        
        {/* A. Top Bar: Diagnostic HUD */}
        <div className="col-span-3 border-b border-slate-800 bg-[#0a0a0a]/90 backdrop-blur-md z-50">
          <DiagnosticContextBar />
        </div>

        {/* B. Left Pane: Navigation Rail */}
        <div className="border-r border-slate-800 bg-[#0a0a0a]">
          <Sidebar />
        </div>

        {/* C. Center: The Simulation Canvas (Monitor Engine) */}
        <main className="relative bg-slate-950/50 flex flex-col min-w-0">
          <div className="absolute inset-0 z-0">
             {/* This is where the WebGL Shader Background would go */}
             <MonitorContainer />
          </div>
        </main>

        {/* D. Right Pane: Remedial Manifold (Collapsible) */}
        {isManifoldOpen && (
          <aside className="w-[400px] border-l border-slate-800 bg-[#0a0a0a] flex flex-col">
            <RemedialManifold />
          </aside>
        )}
      </div>
    </div>
  );
};