import React from "react";
import { useSelection } from '../../lib/context/SelectionContext';
import { ScrollTimeline, TimelineEvent } from "../lightswind/scroll-timeline"; // Assumes file is named scroll-timeline.tsx
import { PlayCircle, AlertOctagon, Wrench, CheckCircle2 } from "lucide-react";

// Mock data generator for the Session
const getSessionEvents = (): TimelineEvent[] => [
  {
    year: "10:42:01",
    title: "Scan Initiated",
    subtitle: "User Triggered",
    description: "Deep scan started on /core/physics/kernels.",
    icon: <PlayCircle className="w-4 h-4 text-blue-400" />,
    color: "blue-400"
  },
  {
    year: "10:42:05",
    title: "Drift Detected",
    subtitle: "Axiom Violation",
    description: "h_norm divergence > 0.5 detected in step 402.",
    icon: <AlertOctagon className="w-4 h-4 text-red-400" />,
    color: "red-400"
  },
  {
    year: "10:42:15",
    title: "Remedial Applied",
    subtitle: "Auto-Fix",
    description: "Applied 'Zero-Init' patch to carry tuple.",
    icon: <Wrench className="w-4 h-4 text-amber-400" />,
    color: "amber-400"
  },
  {
    year: "10:42:18",
    title: "Stabilized",
    subtitle: "System Normal",
    description: "Simulation resumed. Entropy nominal.",
    icon: <CheckCircle2 className="w-4 h-4 text-green-400" />,
    color: "green-400"
  },
];

  const { setSelection } = useSelection();
  const events = getSessionEvents();
  const handleEventClick = (event: TimelineEvent) => {
    setSelection({ type: 'timeline', payload: event });
  };
  return (
    <div className="w-full bg-[#0a0a0a] border-l border-slate-800">
      <div className="p-4 border-b border-slate-800">
        <h3 className="text-sm font-bold text-slate-200 uppercase tracking-widest">Session Timeline</h3>
      </div>
      <div className="h-[calc(100vh-200px)] overflow-y-auto custom-scrollbar">
        {/* Custom rendering to allow click handlers on events */}
        {events.map((event, idx) => (
          <div
            key={idx}
            className="mb-2 cursor-pointer hover:bg-slate-800/40 rounded transition-colors"
            onClick={() => handleEventClick(event)}
          >
            <div className="flex items-center gap-2 px-4 py-2">
              {event.icon}
              <div>
                <div className="font-mono text-xs text-slate-300">{event.title}</div>
                <div className="text-[10px] text-slate-500">{event.subtitle}</div>
                <div className="text-[10px] text-slate-600">{event.year}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};