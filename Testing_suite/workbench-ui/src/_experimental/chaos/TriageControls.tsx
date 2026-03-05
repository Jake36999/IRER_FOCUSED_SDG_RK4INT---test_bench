import React from 'react';
import { ShineButton } from '../lightswind/shine-button'; // Assumes from lightswind
import { ShieldAlert, PowerOff, DatabaseBackup } from 'lucide-react';

export const TriageControls: React.FC = () => {
  return (
    <div className="flex flex-col gap-4 p-6 bg-[#0a0a0a]/90 backdrop-blur border border-red-900/30 rounded-xl shadow-2xl shadow-red-900/10">
      <div className="mb-2">
        <h3 className="text-sm font-black text-slate-200 uppercase tracking-widest mb-1">Override Controls</h3>
        <p className="text-xs text-slate-500 font-mono">Manual intervention required.</p>
      </div>

      {/* Action 1: Hard Kill (Primary Emergency Exit) */}
      <ShineButton 
        className="w-full bg-red-700 hover:bg-red-600 text-white font-bold h-16 flex items-center justify-center gap-3 border border-red-500 transition-all active:scale-95"
        onClick={() => console.log('KILL SIGNAL SENT TO KERNEL')}
      >
        <PowerOff className="w-5 h-5" />
        <span className="tracking-widest text-lg">ABORT KERNEL</span>
      </ShineButton>

      {/* Action 2: Pass to AI Remediation */}
      <button className="w-full bg-amber-600/10 hover:bg-amber-600/20 text-amber-500 font-mono text-sm h-12 flex items-center justify-center gap-2 border border-amber-600/30 rounded transition-all">
        <ShieldAlert className="w-4 h-4" />
        ISOLATE & SEND TO KEL
      </button>

      {/* Action 3: Dump Memory */}
      <button className="w-full bg-slate-800/50 hover:bg-slate-700/50 text-slate-300 font-mono text-sm h-12 flex items-center justify-center gap-2 border border-slate-700 rounded transition-all">
        <DatabaseBackup className="w-4 h-4" />
        DUMP VRAM TRACE
      </button>
      
      <div className="mt-4 pt-4 border-t border-slate-800 text-[10px] text-center text-slate-500 font-mono uppercase">
        Authorization Level: Architect
      </div>
    </div>
  );
};