import React, { useEffect, useState } from 'react';
// You will likely need 'cmdk' installed: npm install cmdk
import { Command } from 'cmdk'; 

export const CommandPalette: React.FC = () => {
  const [open, setOpen] = useState(false);

  // Toggle with Ctrl+K
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((open) => !open);
      }
    };
    document.addEventListener('keydown', down);
    return () => document.removeEventListener('keydown', down);
  }, []);

  return (
    <Command.Dialog
      open={open}
      onOpenChange={setOpen}
      label="Global Command Menu"
      className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm flex items-start justify-center pt-[20vh]"
    >
      <div className="w-full max-w-2xl rounded-xl border border-slate-700 bg-[#0a0a0a] shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-100">
        <div className="flex items-center border-b border-slate-800 px-4">
          <span className="text-slate-500 mr-2">›</span>
          <Command.Input 
            placeholder="Search telemetry, friction points, or run commands..."
            className="flex h-12 w-full rounded-md bg-transparent py-3 text-sm outline-none placeholder:text-slate-500 text-slate-100"
          />
        </div>
        
        <Command.List className="max-h-[300px] overflow-y-auto p-2">
          <Command.Empty className="py-6 text-center text-sm text-slate-500">
            No results found.
          </Command.Empty>

          <Command.Group heading="Actions" className="text-xs font-bold text-slate-500 px-2 py-1.5">
            <Command.Item className="flex items-center gap-2 px-2 py-2 text-sm text-slate-200 rounded cursor-pointer aria-selected:bg-blue-900/30 aria-selected:text-blue-200">
              <span>🚀</span> Run Full Simulation Scan
            </Command.Item>
            <Command.Item className="flex items-center gap-2 px-2 py-2 text-sm text-slate-200 rounded cursor-pointer aria-selected:bg-blue-900/30 aria-selected:text-blue-200">
              <span>⚠️</span> Purge Unindexed Anomalies
            </Command.Item>
          </Command.Group>

          <Command.Group heading="Navigation" className="text-xs font-bold text-slate-500 px-2 py-1.5 mt-2">
            <Command.Item className="px-2 py-2 text-sm text-slate-200 rounded cursor-pointer aria-selected:bg-slate-800">
              Go to Dashboard
            </Command.Item>
            <Command.Item className="px-2 py-2 text-sm text-slate-200 rounded cursor-pointer aria-selected:bg-slate-800">
              Go to Settings
            </Command.Item>
          </Command.Group>
        </Command.List>
      </div>
    </Command.Dialog>
  );
};