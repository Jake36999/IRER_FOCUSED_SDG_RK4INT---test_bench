import { useEffect, useState } from 'react';

export interface KelSummary {
  timestamp: string;
  total_violations: number;
  categories: Record<string, number>;
  top_remedies: [string, number][];
  notable_files: [string, number][];
}

export function useKelSummary() {
  const [summary, setSummary] = useState<KelSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/kel/summary')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch KEL summary');
        return res.json();
      })
      .then((data) => {
        setSummary(data);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, []);

  return { summary, loading, error };
}
