import { useEffect, useState } from 'react';

export function useScanReport() {
  const [fileTree, setFileTree] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/scan/report')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch scan report');
        return res.json();
      })
      .then((data) => {
        setFileTree(data.file_tree || data.files || data);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, []);

  return { fileTree, loading, error };
}
