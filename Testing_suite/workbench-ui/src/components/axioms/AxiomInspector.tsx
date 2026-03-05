import React from "react";
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from "../lightswind/accordion";
import { ShieldCheck, Lock, Cpu } from "lucide-react";

interface Axiom {
  id: string;
  label: string;
  description: string;
  verified?: string;
  icon?: React.ReactNode;
}

interface AxiomInspectorProps {
  axioms?: Axiom[];
}

export const AxiomInspector: React.FC<AxiomInspectorProps> = ({ axioms = [] }) => {
  return (
    <div className="w-full bg-[#0a0a0a] text-slate-300">
      <div className="p-4 border-b border-slate-800 flex justify-between items-center">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Axiom Inspector</h3>
        <span className="text-[10px] bg-green-900/30 text-green-400 px-2 py-0.5 rounded border border-green-800">
          ALL SYSTEMS NOMINAL
        </span>
      </div>

      <Accordion type="multiple" defaultValue={axioms.length ? [axioms[0].id] : []} className="px-4">
        {axioms.map((axiom) => (
          <AccordionItem value={axiom.id} className="border-slate-800" key={axiom.id}>
            <AccordionTrigger className="hover:no-underline hover:text-blue-400 text-sm">
              <div className="flex items-center gap-2">
                {axiom.icon || <ShieldCheck className="w-4 h-4 text-blue-500" />}
                <span>{axiom.label}</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="text-xs text-slate-400 font-mono">
              {axiom.description}
              {axiom.verified && (
                <div className="mt-2 p-2 bg-slate-900 rounded border border-slate-800 text-green-400">
                  ✓ Verified: {axiom.verified}
                </div>
              )}
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  );
};