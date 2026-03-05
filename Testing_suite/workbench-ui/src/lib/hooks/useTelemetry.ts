import { useEffect, useRef, useState } from 'react';


// Telemetry contract (should match backend contract)
export interface TelemetryEvent {
  h_norm_l2: number;
  rho_max?: number;
  drift_vector: number[];
  step: number;
  timestamp: string;
  [key: string]: any;
}


export function useTelemetry(url: string = "/stream") {
  const [events, setEvents] = useState<TelemetryEvent[]>([]);
  const [latest, setLatest] = useState<TelemetryEvent | null>(null);
  const [status, setStatus] = useState<'CONNECTED' | 'DISCONNECTED'>('DISCONNECTED');
  const eventSourceRef = useRef<EventSource | null>(null);

  // Contract validation function
  function validateTelemetryPacket(data: any): data is TelemetryEvent {
    return (
      typeof data === 'object' &&
      typeof data.h_norm_l2 === 'number' &&
      Array.isArray(data.drift_vector) &&
      typeof data.step === 'number' &&
      typeof data.timestamp === 'string'
    );
  }

  useEffect(() => {
    if (!url) return;
    const es = new EventSource(url);
    eventSourceRef.current = es;
    setStatus('CONNECTED');
    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (validateTelemetryPacket(data)) {
          setEvents((prev) => [...prev, data]);
          setLatest(data);
        }
      } catch (e) {
        // Ignore malformed events
      }
    };
    es.onerror = () => {
      setStatus('DISCONNECTED');
      es.close();
    };
    return () => {
      es.close();
    };
  }, [url]);

  return { events, latest, status };
}
