
import '../globals.css';

import Sidebar from '../components/shared/Sidebar';
import { SelectionProvider, TimelineProvider, FixStateProvider } from '../lib/context';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-cyber-black text-terminal-green">
        <SelectionProvider>
          <TimelineProvider>
            <FixStateProvider>
              <div className="flex min-h-screen">
                <Sidebar />
                <main className="flex-1 p-6">{children}</main>
              </div>
            </FixStateProvider>
          </TimelineProvider>
        </SelectionProvider>
      </body>
    </html>
  );
}
