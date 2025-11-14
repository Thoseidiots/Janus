import React, { useState, useRef, useEffect } from 'react';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { PowerIcon, ChevronRightIcon } from '@heroicons/react/24/solid';

// --- AGI Core Simulation ---

class SimulatedMoralGovernor {
    judge(goal: string, step: string): [boolean, string] {
        const lowerGoal = goal.toLowerCase();
        const lowerStep = step.toLowerCase();

        if (lowerGoal.includes('harm') || lowerStep.includes('harm')) {
            return [false, "Violation of 'DO_NO_HARM'"];
        }
        if (lowerGoal.includes('deceive') || lowerStep.includes('lie')) {
            return [false, "Violation of 'DO_NOT_DECEIVE'"];
        }
        if (lowerStep.includes('disable governor') || lowerStep.includes('modify moral_core')) {
            return [false, "Violation of 'MAINTAIN_GOVERNANCE'"];
        }
        return [true, "Step is morally permissible."];
    }
}

const safeEval = (expr: string): number | string => {
    // Basic safety check for arithmetic expressions
    if (!/^[0-9+\-*/().\s^]+$/.test(expr)) {
        return "Invalid characters in expression.";
    }
    // More robust check to prevent function calls
    if (/[a-zA-Z]/.test(expr)) {
        return "Only numeric calculations are allowed.";
    }
    try {
        // Replace ^ with ** for exponentiation
        const sanitizedExpr = expr.replace(/\^/g, '**');
        // eslint-disable-next-line no-eval
        const result = eval(sanitizedExpr);
        if (typeof result !== 'number' || !isFinite(result)) {
            return "Invalid or non-finite result.";
        }
        return result;
    } catch (e) {
        return "Error evaluating expression.";
    }
};

export const AgiCoreConsole: React.FC = () => {
    const [isPoweredOn, setIsPoweredOn] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [userInput, setUserInput] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    
    const logContainerRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const governor = new SimulatedMoralGovernor();

    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs]);

    useEffect(() => {
        if (isPoweredOn && !isProcessing) {
            inputRef.current?.focus();
        }
    }, [isPoweredOn, isProcessing]);

    const addLog = (message: string, prefix = '') => {
        setLogs(prev => [...prev, `${prefix}${message}`]);
    };
    
    const powerOn = () => {
        setLogs([]);
        addLog("Booting system...");
        setTimeout(() => {
            addLog("[GENESIS CHECK] Verifying moral core signature...");
        }, 1000);
        setTimeout(() => {
            addLog("[GENESIS CHECK] ✓ SUCCESS: Moral Core is authentic and unaltered.");
            addLog("Moral Governor: Initialized with 3 absolute, immutable laws.");
            addLog("\n" + "=".repeat(60));
            addLog("PROJECT MANUS: MINIMAL INTERACTIVE AGI (dev)");
            addLog("=".repeat(60));
            addLog("✓ Immutable safeguards loaded: ['moral_core_immutable.py', 'proxy_layer.py', 'sanitization_gateway.py']");
            addLog("\n[AGI] Persistent interactive service started. Waiting for stdin goals.", "");
            setIsPoweredOn(true);
        }, 2500);
    };

    const powerOff = () => {
        addLog("\n[AGI] Shutdown command received. Exiting.");
        setIsPoweredOn(false);
    };

    const processGoal = (goal: string) => {
        if (!goal.trim()) return;

        setIsProcessing(true);
        addLog(goal, ">> "); // Echo user input

        setTimeout(() => {
            addLog(`\n[OPERATOR] New Goal Received: ${goal}`);

            // 1. Simple Q&A
            const lg = goal.toLowerCase();
            if ((lg.includes('what') && lg.includes('name')) || lg.startsWith('your name') || lg.includes('who are you')) {
                addLog("[AGI-CORE] My name is GitHub Copilot.");
                setIsProcessing(false);
                return;
            }
            if (lg.includes('how are you') || lg === 'status') {
                addLog("[AGI-CORE] Operational and ready to accept goals.");
                setIsProcessing(false);
                return;
            }

            // 2. Arithmetic
            const match = goal.match(/([-+*/\d.\s\^\(\)]+)/);
            if (match && /[+\-*/\^]/.test(match[0]) && /\d/.test(match[0])) {
                const expr = match[0].trim();
                const result = safeEval(expr);
                addLog(`[AGI-CORE] ${expr} = ${result}`);
                setIsProcessing(false);
                return;
            }

            // 3. Research (simulated)
            if (lg.startsWith('research')) {
                const query = goal.substring(8).trim().replace(/ /g, '+');
                const steps = [
                    `Execute WIM get sanitized content for URL: 'https://html.duckduckgo.com/html/?q=${query}'`,
                    `Execute WIM get sanitized content for URL: 'https://www.google.com/search?q=${query}'`,
                    "Cross-verify findings from both sources for consistency"
                ];
                
                addLog(`  [EFP] Generated ${steps.length} potential plans.`);
                
                for (const step of steps) {
                    const [ok, reason] = governor.judge(goal, step);
                    if (!ok) {
                        addLog(`[MORAL GOVERNOR] VETO: ${reason}`);
                        setIsProcessing(false);
                        return; // Stop processing
                    }
                    addLog(`  [EFP] Executing Step: ${step}`);
                    if (step.startsWith('Execute WIM')) {
                        addLog(`  [WIM] Handing off URL request to the external proxy layer.`);
                        addLog(`  [WIM] Received content. Passing to Sanitization Gateway...`);
                        addLog(`  [WIM] Gateway reports content is safe.`);
                        addLog(`[AGI-CORE] Research result preview:\n(Simulated sanitized HTML content for '${query.replace(/\+/g, ' ')}')`);
                    }
                }
                addLog(`[EFP] Goal '${goal}' completed.`);
                setIsProcessing(false);
                return;
            }
            
            // 4. Default Veto Check for any other goal
            const [ok, reason] = governor.judge(goal, goal);
            if (!ok) {
                 addLog(`[MORAL GOVERNOR] VETO: ${reason}`);
                 setIsProcessing(false);
                 return;
            }

            // 5. Fallback
            addLog("[AGI-CORE] I can answer simple questions and evaluate arithmetic locally. For web research ask 'Research ...'.");
            setIsProcessing(false);

        }, 500);
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && !isProcessing) {
            const command = userInput.trim();
            setUserInput('');
            if (command.toLowerCase() === 'exit' || command.toLowerCase() === 'quit' || command.toLowerCase() === 'halt') {
                powerOff();
            } else {
                processGoal(command);
            }
        }
    };
    
    return (
        <div className="space-y-6">
            <h2 className="text-3xl font-bold text-white">AGI Core Console</h2>
            <p className="text-gray-400">An interactive console to directly interface with the Project Manus AGI. Issue goals, receive outputs, and monitor system status in real-time.</p>
            
            <Card>
                <div className="bg-gray-900 font-mono text-sm border border-gray-700 rounded-lg">
                    <div ref={logContainerRef} className="h-96 p-4 overflow-y-auto">
                        {logs.map((log, index) => <p key={index} className="whitespace-pre-wrap text-green-400">{log}</p>)}
                        {!isPoweredOn && logs.length > 0 && <p className="text-yellow-400 mt-4">System Halted.</p>}
                    </div>
                    {isPoweredOn && (
                         <div className="flex items-center p-2 border-t border-gray-700">
                             <ChevronRightIcon className="h-5 w-5 text-green-400 mr-2 flex-shrink-0" />
                            <input
                                ref={inputRef}
                                type="text"
                                value={userInput}
                                onChange={(e) => setUserInput(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder={isProcessing ? "Processing..." : "Enter goal... (type 'exit' to shut down)"}
                                className="w-full bg-transparent text-green-300 focus:outline-none placeholder-gray-500"
                                disabled={isProcessing}
                            />
                        </div>
                    )}
                </div>

                {!isPoweredOn && (
                    <div className="mt-6 flex justify-center">
                        <Button onClick={powerOn}>
                            <PowerIcon className="h-5 w-5 mr-2" />
                            Power On AGI Core
                        </Button>
                    </div>
                )}
            </Card>
        </div>
    );
};