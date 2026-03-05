import React, { useEffect } from 'react';
import { EntropyCanvas } from './EntropyCanvas';
import { CatastrophicStream } from './CatastrophicStream';
import { TriageControls } from './TriageControls';
import { AlertTriangle } from 'lucide-react';

export const ChaosContainer: React.FC = () => {
  // Shake effect on mount for physical feedback of a crash
  useEffect(() => {
    document.body.classList.add('animate-[shake_0.5s_ease-in-out]');
    return () => document.body.classList.remove('animate-[shake_0.5s_ease-in-out]');
  }, []);

  return (
    <div className="relative w-full h-full flex flex-col overflow-hidden bg-red-950/20">
      {/* 1. GPU Shader Background (Non-blocking visual indicator of entropy) */}
      <div className="absolute inset-0 z-0 opacity-60 mix-blend-color-dodge">
        <EntropyCanvas intensity={0.95} />
      </div>

      {/* 2. Critical HUD Overlay */}
      <div className="relative z-10 flex flex-col h-full p-6">
        
        {/* Header Alert */}
        <div className="flex items-center gap-4 mb-6 border-b border-red-900/50 pb-4">
          <div className="p-3 bg-red-600/20 rounded-lg animate-pulse border border-red-500/50">
            <AlertTriangle className="w-8 h-8 text-red-500" />
          </div>
          <div>
            <h1 className="text-2xl font-black text-red-500 tracking-widest uppercase shadow-red-500/50 drop-shadow-md">
              Catastrophic Divergence Detected
            </h1>
            <p className="text-red-400/80 font-mono text-sm">
              AXIOM 2 VIOLATION: JAX KERNEL PANIC | SIMULATION INTEGRITY COMPROMISED
            </p>
          </div>
        </div>

        {/* 3. Spllit View: Cascading Errors vs Hard Controls */}
        <div className="flex-1 grid grid-cols-3 gap-6 min-h-0">
          
          {/* Left/Center: The raw, fast-scrolling exception feed */}
          <div className="col-span-2 h-full bg-[#050000]/80 backdrop-blur-md border border-red-900/30 rounded-xl overflow-hidden shadow-2xl shadow-red-900/20">
            <CatastrophicStream />
          </div>

          {/* Right: The Killswitches */}
          <div className="col-span-1 h-full flex flex-col justify-end">
            <TriageControls />
          </div>

        </div>
      </div>
      
      {/* Vignette Overlay to darken edges and focus attention */}
      <div className="absolute inset-0 z-20 pointer-events-none bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-transparent via-transparent to-[#0a0a0a]/90" />
    </div>
  );
};
