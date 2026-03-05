import { useCallback, useState } from 'react';

export interface RemedialQueryResult {
  status: string;
  remedies: string[];
  message?: string;
}

export function useRemedialQuery() {
  const [result, setResult] = useState<RemedialQueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const queryKel = useCallback(async (file_path: string, error_context: string, selection?: string) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch('/api/v1/kel/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path, error_context, selection }),
      });
      if (!res.ok) throw new Error('Remedial query failed');
      const data = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return { result, loading, error, queryKel };
}
