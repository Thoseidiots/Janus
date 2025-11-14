
import { useState, useCallback, useEffect } from 'react';
import { pollVideoOperation } from '../services/geminiService';

export enum VeoStatus {
    IDLE,
    CHECKING_KEY,
    NEEDS_KEY,
    GENERATING,
    SUCCESS,
    ERROR,
}

export const useVeo = () => {
    const [status, setStatus] = useState<VeoStatus>(VeoStatus.IDLE);
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [apiKeySelected, setApiKeySelected] = useState(false);

    const checkApiKey = useCallback(async () => {
        setStatus(VeoStatus.CHECKING_KEY);
        try {
            const hasKey = await window.aistudio.hasSelectedApiKey();
            if (hasKey) {
                setApiKeySelected(true);
                setStatus(VeoStatus.IDLE);
            } else {
                setStatus(VeoStatus.NEEDS_KEY);
            }
        } catch (e) {
            console.error("Error checking API key:", e);
            setStatus(VeoStatus.ERROR);
            setError("Could not verify API key status.");
        }
    }, []);

    useEffect(() => {
        if (typeof window.aistudio !== 'undefined') {
            checkApiKey();
        } else {
            // Handle case where aistudio is not available
             setError("aistudio environment not found. Video generation is disabled.");
             setStatus(VeoStatus.ERROR);
        }
    }, [checkApiKey]);

    const selectApiKey = useCallback(async () => {
        try {
            await window.aistudio.openSelectKey();
            setApiKeySelected(true); // Assume success to avoid race condition
            setStatus(VeoStatus.IDLE);
        } catch (e) {
            console.error("Error opening API key selection:", e);
            setStatus(VeoStatus.ERROR);
            setError("Failed to open API key selection dialog.");
        }
    }, []);

    const handleGeneration = useCallback(async (startGeneration: () => Promise<any>) => {
        setStatus(VeoStatus.GENERATING);
        setVideoUrl(null);
        setError(null);

        try {
            let operation = await startGeneration();
            while (!operation.done) {
                await new Promise(resolve => setTimeout(resolve, 10000));
                operation = await pollVideoOperation(operation);
            }

            const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;
            if (downloadLink) {
                 const response = await fetch(`${downloadLink}&key=${process.env.API_KEY}`);
                 const blob = await response.blob();
                 setVideoUrl(URL.createObjectURL(blob));
                setStatus(VeoStatus.SUCCESS);
            } else {
                throw new Error("Video generation completed but no video URI was found.");
            }
        } catch (e: any) {
            console.error("Video generation failed:", e);
             if (e.message?.includes("Requested entity was not found")) {
                setError("API Key is invalid. Please select a valid key.");
                setApiKeySelected(false);
                setStatus(VeoStatus.NEEDS_KEY);
             } else {
                setError(e.message || "An unknown error occurred during video generation.");
                setStatus(VeoStatus.ERROR);
             }
        }
    }, []);
    
    const reset = useCallback(() => {
        setStatus(VeoStatus.IDLE);
        setVideoUrl(null);
        setError(null);
        if(apiKeySelected) setStatus(VeoStatus.IDLE);
        else setStatus(VeoStatus.NEEDS_KEY);
    }, [apiKeySelected]);

    return { status, videoUrl, error, apiKeySelected, handleGeneration, selectApiKey, reset };
};
