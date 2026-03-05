import React, { forwardRef } from 'react';
import { clsx } from 'clsx'; // Assumes you have clsx or classnames installed

// --- Discriminated Union for Polymorphic Behavior ---
type BaseItemProps = {
  label: string;
  icon: React.ReactNode;
  isActive?: boolean;
  className?: string;
};

type LinkItemProps = BaseItemProps & {
  type: 'link';
  href: string;
};

type ActionItemProps = BaseItemProps & {
  type: 'action';
  onClick: () => void;
  shortcut?: string; // e.g. "⌘K"
};

export type SidebarItemProps = LinkItemProps | ActionItemProps;

export const Sidebar: React.FC = () => {
  return (
    <nav className="flex flex-col gap-2 p-3 h-full">
      <div className="mb-6 px-2 text-xs font-bold text-slate-500 uppercase tracking-widest">
        Sovereign Auditor
      </div>
      
      <SidebarItem 
        type="link" 
        label="Dashboard" 
        href="/dashboard" 
        icon={<span>☷</span>} 
        isActive 
      />
      <SidebarItem 
        type="link" 
        label="Friction Stream" 
        href="/friction" 
        icon={<span>⚡</span>} 
      />
      
      <div className="my-4 h-px bg-slate-800" />
      
      <SidebarItem 
        type="action" 
        label="Run Deep Scan" 
        onClick={() => console.log('Triggering JAX Scan...')} 
        icon={<span>◉</span>} 
        className="text-blue-400 hover:bg-blue-950/30"
      />
    </nav>
  );
};

// --- The Polymorphic Item Component ---
const SidebarItem = forwardRef<HTMLDivElement, SidebarItemProps>((props, ref) => {
  const baseStyles = clsx(
    "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-all duration-200 group cursor-pointer",
    props.isActive 
      ? "bg-slate-800 text-white shadow-[0_0_15px_rgba(255,255,255,0.05)]" 
      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50",
    props.className
  );

  const content = (
    <>
      <span className="opacity-70 group-hover:opacity-100 transition-opacity">{props.icon}</span>
      <span>{props.label}</span>
      {props.type === 'action' && props.shortcut && (
        <span className="ml-auto text-[10px] bg-slate-800 px-1 rounded border border-slate-700 text-slate-500">
          {props.shortcut}
        </span>
      )}
    </>
  );

  if (props.type === 'link') {
    return (
      <a href={props.href} className={baseStyles} ref={ref as any}>
        {content}
      </a>
    );
  }

  return (
    <button onClick={props.onClick} className={clsx(baseStyles, "w-full text-left")} ref={ref as any}>
      {content}
    </button>
  );
});

SidebarItem.displayName = "SidebarItem";