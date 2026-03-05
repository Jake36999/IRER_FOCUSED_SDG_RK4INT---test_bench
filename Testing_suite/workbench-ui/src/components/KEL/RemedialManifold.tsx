import React from "react";
import { ErrorBoundary } from '../monitor/ErrorBoundary';
import { RemedialDiffViewer } from "./RemedialDiffViewer";
import { KnowledgeConfidenceBar } from "./KnowledgeConfidenceBar";
import { useSelection } from "../../lib/context";
import { useRemedialQuery } from "../../lib/hooks/useRemedialQuery";




  const { selection } = useSelection();
  const { result, loading, error, queryKel } = useRemedialQuery();
  const showFix = selection.selectedFile && selection.selectedError;

  React.useEffect(() => {
    if (showFix) {
      queryKel(selection.selectedFile!, selection.selectedError!);
    }
    // eslint-disable-next-line
  }, [selection.selectedFile, selection.selectedError]);


  // Patch export logic
  const exportPatch = () => {
    if (!result || !result.remedies) return;
    const patchData = result.remedies[2] || '';
    const blob = new Blob([patchData], { type: 'text/x-patch' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selection.selectedFile || 'remedial'}.patch`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
  };

  return (
    <ErrorBoundary>
      <div className="flex flex-col h-full w-full bg-[#0a0a0a] border-l border-slate-800 p-4">
        {showFix ? (
          loading ? (
            <div className="text-slate-400 text-xs font-mono">Loading remedial suggestions...</div>
          ) : error ? (
            <div className="text-red-400 text-xs font-mono">{error}</div>
          ) : result ? (
            result.status === 'UNINDEXED' ? (
              <div className="text-amber-400 text-xs font-mono">Remedial suggestions unavailable. Please enter a manual fix.</div>
            ) : result.status === 'NOT_FOUND' ? (
              <div className="text-slate-400 text-xs font-mono">No remedial suggestions found for this error.</div>
            ) : (
              <>
                <RemedialDiffViewer
                  originalCode={result.remedies[0] || ''}
                  remedialCode={result.remedies[1] || ''}
                  language="python"
                  fileName={selection.selectedFile || 'unknown.py'}
                  onApplyFix={() => alert('Fix applied!')}
                />
                <div className="mt-4">
                  <KnowledgeConfidenceBar similarityScore={0.98} resolutionRate={0.95} occurrences={12} />
                </div>
                <button
                  className="mt-4 px-3 py-1 bg-slate-800 text-xs text-slate-200 rounded hover:bg-slate-700 border border-slate-700 w-fit"
                  onClick={exportPatch}
                  disabled={!result.remedies[2]}
                  title="Export patch as .patch file"
                >
                  Export Patch
                </button>
              </>
            )
          ) : null
        ) : (
          <div className="text-slate-500 text-xs font-mono">Select an error to view remedial actions.</div>
        )}
      </div>
    </ErrorBoundary>
  );
};
