import React, { useState, useRef, useEffect, useCallback, FC } from 'react';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { CheckCircleIcon, XCircleIcon, ShieldCheckIcon, RocketLaunchIcon } from '@heroicons/react/24/solid';
import { Spinner } from '../common/Spinner';

type Status = 'idle' | 'pending' | 'success' | 'error';
type LaunchStep = 'idle' | 'vpn' | 'tor' | 'confirm' | 'launched' | 'shuttingdown';

const StatusIndicator: FC<{ status: Status; text: string }> = ({ status, text }) => {
    const getIcon = () => {
        switch (status) {
            case 'pending':
                return <Spinner size="sm" />;
            case 'success':
                return <CheckCircleIcon className="h-6 w-6 text-green-400" />;
            case 'error':
                return <XCircleIcon className="h-6 w-6 text-red-400" />;
            case 'idle':
            default:
                return <div className="h-6 w-6 border-2 border-gray-600 rounded-full" />;
        }
    };
    return (
        <div className="flex items-center gap-3 p-3 bg-gray-700 rounded-lg">
            {getIcon()}
            <span className={`font-medium ${status === 'success' ? 'text-green-300' : 'text-gray-300'}`}>{text}</span>
        </div>
    );
};

export const SecureLauncher: FC = () => {
    const [step, setStep] = useState<LaunchStep>('idle');
    const [logs, setLogs] = useState<string[]>([]);
    const [vpnStatus, setVpnStatus] = useState<Status>('idle');
    const [torStatus, setTorStatus] = useState<Status>('idle');
    const [confirmInput, setConfirmInput] = useState('');
    const logContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs]);

    const addLog = useCallback((message: string) => {
        const timestamp = new Date().toLocaleTimeString();
        setLogs(prev => [...prev, `[${timestamp}] ${message}`]);
    }, []);

    const resetState = useCallback(() => {
        setStep('idle');
        setLogs([]);
        setVpnStatus('idle');
        setTorStatus('idle');
        setConfirmInput('');
    }, []);
    
    const handleLaunch = useCallback(() => {
        resetState();
        setStep('vpn');
        addLog("--- Project Manus Launch Assistant (v2.3) ---");
        
        const simulateTor = () => {
            addLog("[LAUNCHER] Step 2: Starting Tor Service...");
            setTorStatus('pending');
            setTimeout(() => addLog("  > Tor process started in background. Waiting for proxy to become active..."), 1500);
            setTimeout(() => {
                addLog("  > SUCCESS: Tor SOCKS proxy is active on port 9050.");
                setTorStatus('success');
                setStep('confirm');
            }, 4000);
        };

        addLog("[LAUNCHER] Step 1: Connecting to Proton VPN...");
        setVpnStatus('pending');
        setTimeout(() => addLog("  > Public IP before VPN attempt: 1.2.3.4"), 1000);
        setTimeout(() => addLog("  > Launching Proton VPN GUI..."), 2500);
        setTimeout(() => addLog("  > Waiting for ProtonVPN to become active..."), 4000);
        setTimeout(() => {
            addLog("  > DETECT: VPN adapter present (elapsed 5s).");
            addLog("  > Public IP after connect: 5.6.7.8");
            addLog("  > SUCCESS: Public IP changed. VPN Active.");
            setVpnStatus('success');
            setStep('tor');
            simulateTor();
        }, 6000);
    }, [addLog, resetState]);
    
    const handleShutdown = useCallback(() => {
        setStep('shuttingdown');
        addLog("[LAUNCHER] Initiating cleanup sequence...");
        setTimeout(() => {
            addLog("  > Terminating background Tor process...");
            setTorStatus('idle');
        }, 1500);
        setTimeout(() => {
            addLog("  > Disconnecting from Proton VPN...");
            setVpnStatus('idle');
        }, 3000);
        setTimeout(() => {
            addLog("  > Shutdown complete.");
            resetState();
        }, 4500);
    }, [addLog, resetState]);

    const handleConfirm = useCallback(() => {
        if (confirmInput === 'LAUNCH') {
            addLog("[LAUNCHER] Confirmation received. Launching AGI core...");
            setStep('launched');
            setTimeout(() => addLog("[LAUNCHER] 'agi_core_live_v7.py' has been activated in a new process."), 1500);
            setTimeout(() => addLog("[LAUNCHER] The AGI is now running autonomously. This launcher will now exit."), 3000);
        } else {
            addLog("[LAUNCHER] Incorrect confirmation. Launch aborted.");
        }
    }, [addLog, confirmInput]);

    const isBusy = step === 'vpn' || step === 'tor' || step === 'shuttingdown';

    return (
        <div className="space-y-6">
            <h2 className="text-3xl font-bold text-white">Secure Launcher</h2>
            <p className="text-gray-400">Initiate the pre-flight checklist to establish a secure connection before activating the AGI core. This process mirrors the functionality of the `launcher.py` script.</p>

            <Card className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                    <h3 className="text-xl font-semibold">System Status</h3>
                    <div className="space-y-3">
                        <StatusIndicator status={vpnStatus} text="ProtonVPN Connection" />
                        <StatusIndicator status={torStatus} text="Tor Network" />
                    </div>
                    {step === 'launched' && (
                        <div className="p-4 bg-green-900/50 border border-green-700 text-green-300 rounded-lg text-center">
                            <RocketLaunchIcon className="h-12 w-12 mx-auto text-green-400 mb-2"/>
                            <h4 className="text-xl font-bold">AGI Core is Active</h4>
                            <p className="text-sm">System is running autonomously.</p>
                        </div>
                    )}
                    {step === 'confirm' && (
                        <div className="p-4 bg-indigo-900/50 border border-indigo-700 rounded-lg space-y-3">
                            <h4 className="text-lg font-semibold text-indigo-300">Final Launch Confirmation</h4>
                            <p className="text-sm text-gray-400">The AGI is ready for activation. All infrastructure is live. To proceed, type 'LAUNCH' below and confirm.</p>
                            <input
                                type="text"
                                value={confirmInput}
                                onChange={(e) => setConfirmInput(e.target.value)}
                                placeholder="Type 'LAUNCH' to activate"
                                className="w-full p-2 bg-gray-900 border border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500"
                            />
                            <Button onClick={handleConfirm} disabled={confirmInput !== 'LAUNCH'} className="w-full">
                                Activate AGI Core
                            </Button>
                        </div>
                    )}

                    <div className="pt-4 flex items-center gap-4">
                        <Button onClick={handleLaunch} disabled={isBusy || step === 'launched'} isLoading={isBusy}>
                            <ShieldCheckIcon className="h-5 w-5 mr-2" />
                            Initiate Secure Launch
                        </Button>
                         {(step === 'launched' || step === 'confirm') && (
                            <Button variant="secondary" onClick={handleShutdown} disabled={step === 'shuttingdown'}>
                                Shutdown Systems
                            </Button>
                        )}
                    </div>
                </div>
                <div className="space-y-2">
                     <h3 className="text-xl font-semibold">Activity Log</h3>
                    <div ref={logContainerRef} className="h-80 bg-gray-900 text-sm font-mono text-gray-300 p-3 rounded-lg overflow-y-auto border border-gray-700">
                        {logs.length === 0 ? (
                            <span className="text-gray-500">Awaiting system command...</span>
                        ) : (
                            logs.map((log, index) => <p key={index} className="whitespace-pre-wrap">{log}</p>)
                        )}
                    </div>
                </div>
            </Card>
        </div>
    );
};