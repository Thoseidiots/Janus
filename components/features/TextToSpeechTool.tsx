import React, { useState, useRef, useEffect, useCallback, FC } from 'react';
import { generateSpeech } from '../../services/geminiService';
import { decode, decodeAudioData } from '../../utils/audioUtils';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
import { SpeakerWaveIcon, PlayIcon, StopIcon } from '@heroicons/react/24/solid';

export const TextToSpeechTool: FC = () => {
    const [text, setText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    
    const audioContextRef = useRef<AudioContext | null>(null);
    const audioBufferRef = useRef<AudioBuffer | null>(null);
    const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
    
    useEffect(() => {
        // Initialize AudioContext on component mount
        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        }
        
        // Cleanup function to close AudioContext on unmount
        return () => {
            if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
                audioContextRef.current.close();
            }
        };
    }, []);

    const stopPlayback = useCallback(() => {
        if (sourceNodeRef.current) {
            sourceNodeRef.current.stop();
            sourceNodeRef.current.disconnect();
            sourceNodeRef.current = null;
        }
        setIsPlaying(false);
    }, []);

    const playAudio = useCallback(() => {
        if (!audioBufferRef.current || !audioContextRef.current || audioContextRef.current.state === 'closed') return;

        // Stop any previous playback
        if (sourceNodeRef.current) {
             stopPlayback();
        }

        const source = audioContextRef.current.createBufferSource();
        source.buffer = audioBufferRef.current;
        source.connect(audioContextRef.current.destination);
        source.onended = () => {
            setIsPlaying(false);
            sourceNodeRef.current = null;
        };
        source.start(0);
        
        sourceNodeRef.current = source;
        setIsPlaying(true);
    }, [stopPlayback]);

    const handleGenerate = useCallback(async (e: React.FormEvent) => {
        e.preventDefault();
        if (!text || isLoading) return;

        setIsLoading(true);
        setError(null);
        stopPlayback();
        audioBufferRef.current = null;

        try {
            const base64Audio = await generateSpeech(text);
            const decodedAudio = decode(base64Audio);
            if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
                const buffer = await decodeAudioData(decodedAudio, audioContextRef.current, 24000, 1);
                audioBufferRef.current = buffer;
                playAudio(); // Autoplay after generation
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [text, isLoading, stopPlayback, playAudio]);
    
    return (
        <div className="space-y-6">
            <h2 className="text-3xl font-bold text-white">Text-to-Speech</h2>
            <p className="text-gray-400">Convert text, such as model explanations or generated results, into natural-sounding speech.</p>
            
            <Card>
                <form onSubmit={handleGenerate} className="space-y-4">
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="Enter text to convert to speech..."
                        className="w-full h-40 p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                        disabled={isLoading}
                    />
                    <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                       <div className="flex items-center gap-4">
                           <Button type="submit" isLoading={isLoading} disabled={!text}>
                                <SpeakerWaveIcon className="h-5 w-5 mr-2" />
                                Generate & Play
                           </Button>
                           {audioBufferRef.current && (
                               <Button type="button" variant="secondary" onClick={isPlaying ? stopPlayback : playAudio}>
                                   {isPlaying ? <StopIcon className="h-5 w-5 mr-2" /> : <PlayIcon className="h-5 w-5 mr-2" />}
                                   {isPlaying ? 'Stop' : 'Replay'}
                               </Button>
                           )}
                       </div>
                       {isLoading && <Spinner size="sm" />}
                    </div>
                </form>
            </Card>

            {error && (
                <Card>
                    <p className="text-red-400">Error: {error}</p>
                </Card>
            )}

        </div>
    );
};