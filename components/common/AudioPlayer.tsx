
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { decode, decodeAudioData } from '../../utils/audioUtils';
import { Button } from './Button';
import { PlayIcon, StopIcon } from '@heroicons/react/24/solid';

interface AudioPlayerProps {
    base64Audio: string;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({ base64Audio }) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const audioContextRef = useRef<AudioContext | null>(null);
    const audioBufferRef = useRef<AudioBuffer | null>(null);
    const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);

    useEffect(() => {
        const initAudio = async () => {
            try {
                if (!audioContextRef.current) {
                    audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
                }
                const decodedAudio = decode(base64Audio);
                const buffer = await decodeAudioData(decodedAudio, audioContextRef.current, 24000, 1);
                audioBufferRef.current = buffer;
            } catch (err) {
                setError("Failed to decode audio.");
                console.error(err);
            }
        };
        initAudio();
        
        return () => {
            sourceNodeRef.current?.stop();
        };
    }, [base64Audio]);

    const stopPlayback = useCallback(() => {
        if (sourceNodeRef.current) {
            sourceNodeRef.current.stop();
            sourceNodeRef.current.disconnect();
            sourceNodeRef.current = null;
        }
        setIsPlaying(false);
    }, []);

    const playAudio = useCallback(() => {
        if (!audioBufferRef.current || !audioContextRef.current || isPlaying) return;

        stopPlayback();

        const source = audioContextRef.current.createBufferSource();
        source.buffer = audioBufferRef.current;
        source.connect(audioContextRef.current.destination);
        source.onended = () => setIsPlaying(false);
        source.start(0);
        
        sourceNodeRef.current = source;
        setIsPlaying(true);
    }, [isPlaying, stopPlayback]);

    if (error) {
        return <p className="text-red-400 text-sm">{error}</p>;
    }

    return (
        <Button variant="secondary" onClick={isPlaying ? stopPlayback : playAudio} disabled={!audioBufferRef.current}>
            {isPlaying ? <StopIcon className="h-5 w-5 mr-2" /> : <PlayIcon className="h-5 w-5 mr-2" />}
            {isPlaying ? 'Stop' : 'Play Audio'}
        </Button>
    );
};
