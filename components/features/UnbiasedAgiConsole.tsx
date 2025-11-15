import React, { useState, useRef, useEffect } from 'react';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
// Fix: Import `PowerIcon` to resolve 'Cannot find name' error.
import { CpuChipIcon, BeakerIcon, LightBulbIcon, PowerIcon } from '@heroicons/react/24/solid';

// Fix: Change return type to `React.ReactElement` to resolve 'Cannot find namespace JSX' error.
const colorizeLog = (log: string): React.ReactElement => {
    const colorMap: { [key: string]: string } = {
        '[EFP]': 'text-yellow-400', // Executive Function & Planning
        '[WIM]': 'text-cyan-400',   // Web Interface Module
        '[CSR]': 'text-purple-400', // Common Sense Reasoning
        '[CRE]': 'text-orange-400', // Core Reasoning Engine
        '[LTM]': 'text-blue-400',   // Long-Term Memory
        '[LEARN]': 'text-green-300 font-bold', // Learning
        '[META]': 'text-pink-400 font-bold', // Meta-cognition
        'SUCCESS': 'text-green-400',
        '✓': 'text-green-400',
    };

    for (const prefix in colorMap) {
        if (log.includes(prefix)) {
            return <p className={`whitespace-pre-wrap ${colorMap[prefix]}`}>{log}</p>;
        }
    }
    return <p className="whitespace-pre-wrap">{log}</p>;
};

export const UnbiasedAgiConsole: React.FC = () => {
    const [isPoweredOn, setIsPoweredOn] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [isBusy, setIsBusy] = useState(false);
    const [ltmSize, setLtmSize] = useState(2);
    const [creFacts, setCreFacts] = useState(3);
    
    const logContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs]);

    const addLog = (message: string) => {
        setLogs(prev => [...prev, message]);
    };

    const runSimulation = (goal: string, onComplete: () => void) => {
        let logQueue: { msg: string, delay: number }[] = [];

        const plan = goal.toLowerCase().includes("research")
            ? ["Search multiple sources", "Cross-reference findings", "Detect bias", "Synthesize consensus", "Store findings"]
            : [`Step 1: ${goal}`, `Step 2: Execute`];

        logQueue.push({ msg: `\n[EFP] Setting Goal: ${goal}`, delay: 100 });
        
        plan.forEach((step, index) => {
            logQueue.push({ msg: `\n[EFP] Executing Step: ${step}`, delay: 500 * (index + 1) });
            if (step.toLowerCase().includes("search")) {
                logQueue.push({ msg: "  [EFP] Multi-source research required.", delay: 500 * (index + 1) + 100 });
                logQueue.push({ msg: "  [WIM] Searching all 4 engines...", delay: 500 * (index + 1) + 600 });
                logQueue.push({ msg: "  [WIM] Research completed: Consensus score: 0.85, Bias risk: 0.15", delay: 500 * (index + 1) + 1200 });
            }
            if (step.toLowerCase().includes("store")) {
                logQueue.push({ msg: "  [LEARN] Stored findings in LTM and CRE.", delay: 500 * (index + 1) + 200 });
            }
        });

        logQueue.push({ msg: `\n[EFP] Goal '${goal}' successfully completed.`, delay: 500 * plan.length + 500 });

        let cumulativeDelay = 0;
        logQueue.forEach(({ msg, delay }) => {
            cumulativeDelay += delay;
            setTimeout(() => addLog(msg), cumulativeDelay);
        });

        setTimeout(() => {
            onComplete();
        }, cumulativeDelay + 500);
    };

    const handleRunResearch = () => {
        setIsBusy(true);
        addLog("\n" + "=".repeat(80));
        addLog("TASK: Research the current state of artificial intelligence ethics");
        addLog("=".repeat(80));
        runSimulation("Research the current state of artificial intelligence ethics", () => {
            setLtmSize(prev => prev + 1);
            setCreFacts(prev => prev + 1);
            setIsBusy(false);
        });
    };
    
    const handleCognitiveEnhancement = () => {
        setIsBusy(true);
        addLog("\n" + "=".repeat(80));
        addLog("TASK: Initiate Cognitive Enhancement Cycle");
        addLog("=".repeat(80));

        let enhancementLogs: { msg: string, delay: number }[] = [
            { msg: "[META] Initiating cognitive enhancement cycle.", delay: 500 },
            { msg: "[META] Analyzing Long-Term Memory for knowledge gaps...", delay: 1500 },
            { msg: "[META] Analysis complete. Identified knowledge deficit in 'Neural Architecture Search'.", delay: 1500 },
            { msg: "[META] Self-generating goal: \"Research the current state of Neural Architecture Search (NAS) techniques\".", delay: 1000 },
        ];

        let cumulativeDelay = 0;
        enhancementLogs.forEach(({ msg, delay }) => {
            cumulativeDelay += delay;
            setTimeout(() => addLog(msg), cumulativeDelay);
        });

        setTimeout(() => {
            runSimulation("Research the current state of Neural Architecture Search (NAS) techniques", () => {
                 setTimeout(() => {
                    addLog("[META] Cognitive enhancement cycle complete. Knowledge base has been expanded.");
                    setLtmSize(prev => prev + 1);
                    setCreFacts(prev => prev + 1);
                    setIsBusy(false);
                }, 500);
            });
        }, cumulativeDelay + 500);
    };

    const powerOn = () => {
        setLogs([]);
        addLog("=".repeat(80));
        addLog("PROJECT MANUS: FINAL AGI CORE - UNBIASED INTELLIGENCE");
        addLog("=".repeat(80));
        addLog("✓ Chromium Browser (Blink Engine)");
        addLog("✓ Brave Browser (Blink Engine, Privacy-Focused)");
        addLog("✓ Firefox Browser (Gecko Engine)");
        addLog("✓ Google Search Engine");
        addLog("✓ Bing Search Engine");
        addLog("✓ DuckDuckGo Search Engine (Privacy-Focused)");
        addLog("✓ Yahoo Search Engine");
        addLog(`\nInitial LTM size: ${ltmSize} memories.`);
        addLog(`Initial CRE facts: ${creFacts} facts.`);
        setIsPoweredOn(true);
    };

    if (!isPoweredOn) {
         return (
            <div className="text-center space-y-6">
                 <h2 className="text-3xl font-bold text-white">Unbiased AGI Console</h2>
                <p className="text-gray-400 max-w-2xl mx-auto">The final, integrated AGI system with multi-browser, multi-search-engine support designed for unbiased, comprehensive intelligence and self-improvement.</p>
                <Card className="max-w-md mx-auto">
                    <Button onClick={powerOn}>
                        <PowerIcon className="h-5 w-5 mr-2" />
                        Power On Unbiased AGI
                    </Button>
                </Card>
            </div>
        );
    }
    
    return (
        <div className="space-y-6">
            <h2 className="text-3xl font-bold text-white">Unbiased AGI Console</h2>
            <p className="text-gray-400">Directly interface with the final AGI core. Initiate research tasks or trigger a self-improvement cycle to observe its learning process.</p>
            
            <Card>
                <div className="flex flex-col sm:flex-row justify-between items-center mb-4 gap-4">
                     <div className="flex items-center gap-6 text-sm text-gray-300">
                        <div className="flex items-center gap-2">
                           <CpuChipIcon className="h-6 w-6 text-blue-400" />
                           <span>LTM Size: <span className="font-bold text-white">{ltmSize}</span></span>
                        </div>
                        <div className="flex items-center gap-2">
                           <LightBulbIcon className="h-6 w-6 text-yellow-400" />
                           <span>CRE Facts: <span className="font-bold text-white">{creFacts}</span></span>
                        </div>
                    </div>
                    <div className="flex items-center gap-4">
                        <Button onClick={handleRunResearch} disabled={isBusy} isLoading={isBusy}>
                            <BeakerIcon className="h-5 w-5 mr-2" />
                            Run Research Task
                        </Button>
                        <Button onClick={handleCognitiveEnhancement} disabled={isBusy} isLoading={isBusy} variant="secondary">
                            Initiate Cognitive Enhancement
                        </Button>
                    </div>
                </div>

                <div className="bg-gray-900 font-mono text-sm border border-gray-700 rounded-lg">
                    <div ref={logContainerRef} className="h-96 p-4 overflow-y-auto">
                        {logs.map((log, index) => <div key={index}>{colorizeLog(log)}</div>)}
                    </div>
                </div>
            </Card>
        </div>
    );
};