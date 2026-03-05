import React from "react";
import { Progress } from "../lightswind/progress"; // Assumes progress.tsx is in lightswind folder
import { ShieldCheck, ShieldAlert, Shield } from "lucide-react";

interface KnowledgeConfidenceBarProps {
  similarityScore: number; // 0 to 1
  resolutionRate: number; // 0 to 1
  occurrences: number;
}

export const KnowledgeConfidenceBar: React.FC<KnowledgeConfidenceBarProps> = ({
  similarityScore,
  resolutionRate,
  occurrences
}) => {
  // Determine color and icon based on score
  const getStatus = (score: number) => {
    if (score >= 0.9) return { color: "success", icon: ShieldCheck, label: "Golden Standard", text: "text-green-400" };
    if (score >= 0.7) return { color: "warning", icon: Shield, label: "Probable Fix", text: "text-amber-400" };
    return { color: "danger", icon: ShieldAlert, label: "Low Confidence", text: "text-red-400" };
  };

  const status = getStatus(similarityScore);
  const StatusIcon = status.icon;

  return (
    <div className="w-full p-4 bg-slate-900/50 border border-slate-800 rounded-lg backdrop-blur-sm">
      <div className="flex justify-between items-end mb-2">
        <div className="flex flex-col">
          <span className="text-[10px] uppercase tracking-wider text-slate-500 font-bold">KEL Confidence</span>
          <div className={`flex items-center gap-2 text-sm font-mono font-bold ${status.text}`}>
            <StatusIcon className="w-4 h-4" />
            {status.label} ({(similarityScore * 100).toFixed(1)}%)
          </div>
        </div>
        <div className="text-right">
           <span className="text-[10px] text-slate-500 block">Resolution Rate</span>
           <span className="text-xs text-slate-200 font-mono">{(resolutionRate * 100).toFixed(0)}% ({occurrences} events)</span>
        </div>
      </div>

      <Progress 
        value={similarityScore * 100} 
        max={100}
        // @ts-ignore - The provided component uses string literals for color
        color={status.color}
        size="sm"
        className="bg-slate-800"
        indicatorClassName="bg-gradient-to-r from-transparent via-current to-current"
      />
    </div>
  );
};