import React, { useState, useCallback } from 'react';
import { Card } from '../../common/Card';
import { Button } from '../../common/Button';
import { ShieldCheckIcon, RocketLaunchIcon } from '@heroicons/react/24/solid';

interface SecureLauncherViewProps {
    onLaunch: () => void;
}

export const SecureLauncherView: React.FC<SecureLauncherViewProps> = ({ onLaunch }) => {
    const [step, setStep] = useState<'idle' | 'vpn' | 'tor' | 'confirm'>('idle');
    const [logs, setLogs] = useState<string[]>([]);
    
    const addLog = useCallback((message: string) => {
        setLogs(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] ${message}`]);
    }, []);

    const startLaunch = useCallback(() => {
        setStep('vpn');
        addLog("Launch sequence initiated...");
        setTimeout(() => {
            addLog("Secure VPN Tunnel... [ESTABLISHED]");
            setStep('tor');
            setTimeout(() => {
                addLog("Tor Network... [ACTIVE]");
                setStep('confirm');
            }, 1500);
        }, 1500);
    }, [addLog]);

    return (
        <div className="relative flex items-center justify-center h-full p-4 animate-fadeIn scanline-overlay">
            <Card className="max-w-2xl text-center bg-gray-900/80 backdrop-blur-sm border-gray-700">
                <ShieldCheckIcon className="h-16 w-16 mx-auto text-indigo-400 mb-4" />
                <h2 className="text-3xl font-bold mb-2">Janus AGI Control</h2>
                <p className="text-gray-400 mb-6">System is offline. Initiate the secure launch sequence to activate Janus.</p>
                
                <div className="text-left font-mono text-xs text-green-400 my-4 bg-black/50 p-4 rounded-lg h-40 overflow-y-auto border border-gray-700">
                    {logs.length === 0 ? <p className="text-gray-500">Awaiting system command...</p> : logs.map((l,i) => <p key={i} className="animate-fadeIn" style={{animationDelay: `${i*50}ms`}}>{l}</p>)}
                </div>

                {step === 'confirm' ? (
                    <div className="animate-fadeIn" style={{animationDelay: '500ms'}}>
                        <p className="text-green-300 mb-4 font-semibold">All systems ready. Awaiting final activation.</p>
                        <Button onClick={onLaunch} className="w-full text-lg py-3">
                            <RocketLaunchIcon className="h-6 w-6 mr-2" />
                            Activate Janus
                        </Button>
                    </div>
                ) : (
                    <Button onClick={startLaunch} isLoading={step === 'vpn' || step === 'tor'} disabled={step !== 'idle'} className="w-full text-lg py-3">
                        <ShieldCheckIcon className="h-6 w-6 mr-2" />
                        Initiate Secure Launch
                    </Button>
                )}
            </Card>
        </div>
    );
};