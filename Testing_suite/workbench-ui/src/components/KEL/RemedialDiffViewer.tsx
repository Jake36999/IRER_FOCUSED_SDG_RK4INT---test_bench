import React, { useState } from "react";
import { Terminal, ArrowRight, Check, AlertTriangle, Play } from "lucide-react";
import { cn } from "../../lib/utils"; // Ensure this path matches your project
import { motion } from "framer-motion";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/cjs/styles/prism";

interface RemedialDiffViewerProps {
  originalCode: string;
  remedialCode: string;
  language?: string;
  fileName?: string;
  onApplyFix?: () => void;
}

export const RemedialDiffViewer: React.FC<RemedialDiffViewerProps> = ({
  originalCode,
  remedialCode,
  language = "python",
  fileName = "kernel_v4.py",
  onApplyFix
}) => {
  const [isApplying, setIsApplying] = useState(false);

  const handleApply = () => {
    setIsApplying(true);
    // Simulate API delay
    setTimeout(() => {
      setIsApplying(false);
      onApplyFix?.();
    }, 1500);
  };

  return (
    <div className="flex flex-col gap-4 w-full h-full max-h-[600px]">
      {/* Header / Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-slate-900/80 backdrop-blur border-b border-slate-800 rounded-t-lg">
        <div className="flex items-center gap-3">
          <Terminal className="w-4 h-4 text-purple-400" />
          <span className="text-xs font-mono text-slate-400">{fileName}</span>
          <span className="px-2 py-0.5 text-[10px] uppercase bg-red-900/30 text-red-400 border border-red-800 rounded">
            Drift Detected
          </span>
        </div>
        <div className="text-xs text-slate-500 font-mono">
          JAX Primitive Correction
        </div>
      </div>

      {/* Diff Container */}
      <div className="grid grid-cols-2 gap-px bg-slate-800 border border-slate-800 overflow-hidden rounded-b-lg flex-1">
        
        {/* Left Pane: Broken Code */}
        <div className="relative bg-[#0a0a0a] overflow-auto custom-scrollbar group">
          <div className="sticky top-0 z-10 flex items-center gap-2 bg-red-950/20 px-3 py-1 text-xs text-red-400 border-b border-red-900/30 backdrop-blur-sm">
            <AlertTriangle className="w-3 h-3" /> Current State
          </div>
          <div className="p-0 text-xs font-mono opacity-80 group-hover:opacity-100 transition-opacity">
            <SyntaxHighlighter
              language={language}
              style={oneDark}
              customStyle={{ background: "transparent", margin: 0, padding: "1rem" }}
              wrapLines
              lineProps={(lineNumber) => ({
                style: { display: "block", backgroundColor: "rgba(239, 68, 68, 0.05)" } 
              })}
            >
              {originalCode}
            </SyntaxHighlighter>
          </div>
        </div>

        {/* Right Pane: Remedial Fix */}
        <div className="relative bg-[#0a0a0a] overflow-auto custom-scrollbar group">
          <div className="sticky top-0 z-10 flex items-center gap-2 bg-green-950/20 px-3 py-1 text-xs text-green-400 border-b border-green-900/30 backdrop-blur-sm">
            <Check className="w-3 h-3" /> KEL Suggestion (Similarity: 0.92)
          </div>
          <div className="p-0 text-xs font-mono opacity-80 group-hover:opacity-100 transition-opacity">
            <SyntaxHighlighter
              language={language}
              style={oneDark}
              customStyle={{ background: "transparent", margin: 0, padding: "1rem" }}
              wrapLines
              lineProps={(lineNumber) => ({
                style: { display: "block", backgroundColor: "rgba(34, 197, 94, 0.05)" }
              })}
            >
              {remedialCode}
            </SyntaxHighlighter>
          </div>
        </div>
      </div>

      {/* Action Footer */}
      <div className="flex justify-end p-2">
        <button
          onClick={handleApply}
          disabled={isApplying}
          className={cn(
            "relative flex items-center gap-2 px-6 py-2 text-sm font-bold text-white transition-all rounded shadow-lg overflow-hidden group",
            isApplying ? "bg-slate-700 cursor-wait" : "bg-blue-600 hover:bg-blue-500 hover:shadow-blue-500/25"
          )}
        >
          {isApplying ? (
            <>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full"
              />
              <span>Refactoring...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-white" />
              <span>Apply Golden Fix</span>
            </>
          )}
          
          {/* Shine Effect from lightswind */}
          <div className="absolute inset-0 -translate-x-full group-hover:animate-[shimmer_1.5s_infinite] bg-gradient-to-r from-transparent via-white/10 to-transparent z-10" />
        </button>
      </div>
    </div>
  );
};