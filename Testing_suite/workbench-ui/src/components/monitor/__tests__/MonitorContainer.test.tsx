import { render, screen } from '@testing-library/react';
import { MonitorContainer } from '../MonitorContainer';

jest.mock('@/lib/hooks/useScanReport', () => ({
  useScanReport: () => ({ fileTree: null, loading: true, error: null })
}));
jest.mock('@/lib/hooks/useTelemetry', () => ({
  useTelemetry: () => ({ events: [], latest: {}, status: 'CONNECTED' })
}));
jest.mock('@/lib/hooks/useKelSummary', () => ({
  useKelSummary: () => ({ summary: { total_vectors: 42 } })
}));

describe('MonitorContainer', () => {
  afterEach(() => {
    jest.resetAllMocks();
  });

  it('renders loading state', () => {
    (require('@/lib/hooks/useScanReport').useScanReport as jest.Mock).mockReturnValueOnce({ fileTree: null, loading: true, error: null });
    render(<MonitorContainer />);
    expect(screen.getByText(/hydrating state/i)).toBeInTheDocument();
  });

  it('renders error state', () => {
    (require('@/lib/hooks/useScanReport').useScanReport as jest.Mock).mockReturnValueOnce({ fileTree: null, loading: false, error: 'Test error' });
    render(<MonitorContainer />);
    expect(screen.getByText(/test error/i)).toBeInTheDocument();
  });

  it('renders main UI after hydration', () => {
    (require('@/lib/hooks/useScanReport').useScanReport as jest.Mock).mockReturnValueOnce({ fileTree: {}, loading: false, error: null });
    render(<MonitorContainer />);
    expect(screen.getByText(/KEL INDEX/i)).toBeInTheDocument();
  });
});
