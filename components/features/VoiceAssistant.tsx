import React, { useState, useRef, useEffect, useCallback, FC } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Blob } from '@google/genai';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { encode, decode, decodeAudioData } from '../../utils/audioUtils';
import { MicrophoneIcon, StopCircleIcon, ChatBubbleBottomCenterTextIcon } from '@heroicons/react/24/solid';

type TranscriptionEntry = {
    speaker: 'user' | 'model';
    text: string;
};

export const VoiceAssistant: FC = () => {
    const [isListening, setIsListening] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [transcriptions, setTranscriptions] = useState<TranscriptionEntry[]>([]);

    const sessionPromiseRef = useRef<Promise<any> | null>(null);
    const inputAudioContextRef = useRef<AudioContext | null>(null);
    const outputAudioContextRef = useRef<AudioContext | null>(null);
    const mediaStreamRef = useRef<MediaStream | null>(null);
    const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
    const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

    const currentInputTranscriptionRef = useRef('');
    const currentOutputTranscriptionRef = useRef('');
    
    const nextStartTimeRef = useRef(0);
    const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

    const cleanup = useCallback(() => {
        if (sessionPromiseRef.current) {
            sessionPromiseRef.current.then(session => session.close()).catch(console.error);
            sessionPromiseRef.current = null;
        }
        
        scriptProcessorRef.current?.disconnect();
        mediaStreamSourceRef.current?.disconnect();
        scriptProcessorRef.current = null;
        mediaStreamSourceRef.current = null;
        
        mediaStreamRef.current?.getTracks().forEach(track => track.stop());
        mediaStreamRef.current = null;
        
        if (inputAudioContextRef.current?.state !== 'closed') {
            inputAudioContextRef.current?.close().catch(console.error);
        }
        if (outputAudioContextRef.current?.state !== 'closed') {
            outputAudioContextRef.current?.close().catch(console.error);
        }
        inputAudioContextRef.current = null;
        outputAudioContextRef.current = null;
        
        setIsListening(false);
    }, []);

    useEffect(() => {
        return () => cleanup();
    }, [cleanup]);

    const startConversation = useCallback(async () => {
        if (isListening) return;
        setError(null);
        setTranscriptions([]);
        currentInputTranscriptionRef.current = '';
        currentOutputTranscriptionRef.current = '';
        setIsListening(true);

        try {
            if (!process.env.API_KEY) throw new Error("API Key not found.");
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            
            inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
            outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
            nextStartTimeRef.current = 0;
            
            sessionPromiseRef.current = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                callbacks: {
                    onopen: async () => {
                        console.log("Session opened.");
                        try {
                            mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
                            const source = inputAudioContextRef.current!.createMediaStreamSource(mediaStreamRef.current);
                            mediaStreamSourceRef.current = source;
                            const scriptProcessor = inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
                            scriptProcessorRef.current = scriptProcessor;

                            scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
                                const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                                const pcmBlob: Blob = {
                                    data: encode(new Uint8Array(new Int16Array(inputData.map(v => v * 32768)).buffer)),
                                    mimeType: 'audio/pcm;rate=16000',
                                };
                                if (sessionPromiseRef.current) {
                                    sessionPromiseRef.current.then((session) => {
                                        session.sendRealtimeInput({ media: pcmBlob });
                                    });
                                }
                            };
                            source.connect(scriptProcessor);
                            scriptProcessor.connect(inputAudioContextRef.current!.destination);
                        } catch (err) {
                             console.error('Error accessing microphone:', err);
                             setError(err instanceof Error ? `Microphone access error: ${err.message}` : 'Could not access microphone.');
                             cleanup();
                        }
                    },
                    onmessage: async (message: LiveServerMessage) => {
                        if (message.serverContent?.inputTranscription) {
                            currentInputTranscriptionRef.current += message.serverContent.inputTranscription.text;
                        }
                        if (message.serverContent?.outputTranscription) {
                            currentOutputTranscriptionRef.current += message.serverContent.outputTranscription.text;
                        }
                        if (message.serverContent?.turnComplete) {
                            const fullInput = currentInputTranscriptionRef.current.trim();
                            const fullOutput = currentOutputTranscriptionRef.current.trim();
                            setTranscriptions(prev => {
                                const newEntries: TranscriptionEntry[] = [];
                                if (fullInput) newEntries.push({ speaker: 'user', text: fullInput });
                                if (fullOutput) newEntries.push({ speaker: 'model', text: fullOutput });
                                return [...prev, ...newEntries];
                            });
                            currentInputTranscriptionRef.current = '';
                            currentOutputTranscriptionRef.current = '';
                        }
                        
                        const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                        if (base64Audio && outputAudioContextRef.current) {
                            nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputAudioContextRef.current.currentTime);
                            const audioBuffer = await decodeAudioData(decode(base64Audio), outputAudioContextRef.current, 24000, 1);
                            const source = outputAudioContextRef.current.createBufferSource();
                            source.buffer = audioBuffer;
                            source.connect(outputAudioContextRef.current.destination);
                            source.addEventListener('ended', () => { sourcesRef.current.delete(source); });
                            source.start(nextStartTimeRef.current);
                            nextStartTimeRef.current += audioBuffer.duration;
                            sourcesRef.current.add(source);
                        }
                        
                        if (message.serverContent?.interrupted) {
                            for (const source of sourcesRef.current.values()) source.stop();
                            sourcesRef.current.clear();
                            nextStartTimeRef.current = 0;
                        }
                    },
                    onerror: (e: ErrorEvent) => {
                        console.error('Session error:', e);
                        setError(`Session error: ${e.message}`);
                        cleanup();
                    },
                    onclose: (e: CloseEvent) => {
                        console.log("Session closed.");
                        cleanup();
                    },
                },
                config: {
                    responseModalities: [Modality.AUDIO],
                    inputAudioTranscription: {},
                    outputAudioTranscription: {},
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
                    systemInstruction: 'You are a friendly and helpful assistant for machine learning tasks.',
                },
            });

        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
            cleanup();
        }
    }, [isListening, cleanup]);

    const stopConversation = useCallback(() => {
        cleanup();
    }, [cleanup]);

    return (
        <div className="space-y-6">
            <h2 className="text-3xl font-bold text-white">Voice Assistant</h2>
            <p className="text-gray-400">Start a real-time conversation with Gemini. Ask questions about hyperparameters, describe your model, and get spoken responses instantly.</p>
            <Card>
                <div className="flex flex-col items-center justify-center space-y-6">
                    <div className="text-center">
                        {!isListening ? (
                            <Button onClick={startConversation} className="p-4 rounded-full">
                                <MicrophoneIcon className="h-8 w-8" />
                            </Button>
                        ) : (
                            <Button onClick={stopConversation} variant="secondary" className="p-4 rounded-full bg-red-600 hover:bg-red-500">
                                <StopCircleIcon className="h-8 w-8" />
                            </Button>
                        )}
                        <p className="mt-2 text-lg font-medium">{isListening ? 'Listening...' : 'Tap to Start'}</p>
                    </div>

                    {error && <p className="text-red-400">Error: {error}</p>}

                    <div className="w-full max-w-2xl h-64 bg-gray-900 rounded-lg p-4 overflow-y-auto border border-gray-700">
                       {transcriptions.length === 0 && !isListening && (
                           <div className="flex flex-col items-center justify-center h-full text-gray-500">
                               <ChatBubbleBottomCenterTextIcon className="h-12 w-12 mb-2"/>
                               <p>Conversation transcript will appear here.</p>
                           </div>
                       )}
                        {transcriptions.map((entry, index) => (
                            <div key={index} className={`mb-2 ${entry.speaker === 'user' ? 'text-right' : 'text-left'}`}>
                                <span className={`inline-block px-3 py-1 rounded-lg ${entry.speaker === 'user' ? 'bg-indigo-600 text-white' : 'bg-gray-700'}`}>
                                    <span className="font-bold capitalize">{entry.speaker}: </span>{entry.text}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            </Card>
        </div>
    );
};