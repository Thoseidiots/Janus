import { GoogleGenAI, GenerateContentResponse, Chat, Type, Modality, Content } from "@google/genai";
import { GroundingSource, ChatMessage, Storyboard } from '../types';

let ai: GoogleGenAI | null = null;
const getAI = () => {
    if (!ai) {
        if (!process.env.API_KEY) {
            throw new Error("API_KEY environment variable not set");
        }
        ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    }
    return ai;
};

// For Veo which requires dynamic API key selection
const getVeoAI = () => {
    if (!process.env.API_KEY) {
        throw new Error("API_KEY environment variable not set for Veo");
    }
    return new GoogleGenAI({ apiKey: process.env.API_KEY });
};

// --- Text Generation ---

export const generateText = async (prompt: string, model: 'gemini-2.5-pro' | 'gemini-2.5-flash' | 'gemini-2.5-flash-lite', useThinking: boolean = false): Promise<string> => {
    const aiInstance = getAI();
    const config = useThinking ? { thinkingConfig: { thinkingBudget: 32768 } } : {};
    
    const response = await aiInstance.models.generateContent({
        model: model,
        contents: prompt,
        config: config
    });
    return response.text;
};

// --- Chat ---

export const generateChatResponse = async (
    history: ChatMessage[],
    newMessage: string,
    useGoogleSearch: boolean
): Promise<{ text: string; sources: GroundingSource[] }> => {
    const aiInstance = getAI();
    const systemInstruction = 'You are a helpful assistant specializing in machine learning hyperparameter tuning. When web search is enabled, you have access to real-time information from Google Search to answer questions about the latest research and developments.';

    const contents: Content[] = history.map(msg => ({
        role: msg.role,
        parts: [{ text: msg.content }]
    }));
    contents.push({ role: 'user', parts: [{ text: newMessage }] });

    const config: any = useGoogleSearch ? { tools: [{ googleSearch: {} }] } : {};

    const response = await aiInstance.models.generateContent({
        model: "gemini-2.5-flash",
        contents: contents,
        config: {
            ...config,
            systemInstruction
        }
    });

    let sources: GroundingSource[] = [];
    if (useGoogleSearch) {
        const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
        sources = groundingChunks
            .map((chunk: any) => ({
                uri: chunk.web?.uri,
                title: chunk.web?.title,
            }))
            .filter((s: any) => s.uri && s.title);
    }

    return { text: response.text, sources };
};

// --- Grounded Search ---

export const generateWithGoogleSearch = async (prompt: string): Promise<{ text: string, sources: GroundingSource[] }> => {
    const aiInstance = getAI();
    const response = await aiInstance.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
        config: { tools: [{ googleSearch: {} }] },
    });
    
    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    const sources = groundingChunks
        .map((chunk: any) => ({
            uri: chunk.web?.uri,
            title: chunk.web?.title,
        }))
        .filter((s: any) => s.uri && s.title);

    return { text: response.text, sources };
};

export const generateWithGoogleMaps = async (prompt: string, latitude?: number, longitude?: number): Promise<{ text: string, sources: GroundingSource[] }> => {
    const aiInstance = getAI();
    const toolConfig: any = {};
    if (latitude && longitude) {
        toolConfig.retrievalConfig = { latLng: { latitude, longitude } };
    }

    const response = await aiInstance.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
        config: { 
            tools: [{ googleMaps: {} }],
        },
        ...(Object.keys(toolConfig).length > 0 && { toolConfig }),
    });

    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    const sources = groundingChunks
        .map((chunk: any) => ({
            uri: chunk.maps?.uri,
            title: chunk.maps?.title,
        }))
        .filter((s: any) => s.uri && s.title);

    return { text: response.text, sources };
};

// --- Image Analysis ---

export const analyzeImage = async (prompt: string, imageBase64: string, mimeType: string): Promise<string> => {
    const aiInstance = getAI();
    const imagePart = { inlineData: { data: imageBase64, mimeType } };
    const textPart = { text: prompt };

    const response = await aiInstance.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: { parts: [textPart, imagePart] },
    });
    return response.text;
};


// --- Video Analysis ---

export const analyzeVideo = async (prompt: string, videoFile: File): Promise<string> => {
    // Note: This is a placeholder for a real video analysis implementation.
    // The Gemini API currently supports video understanding via file URIs, which is not directly usable in a browser upload scenario without a backend.
    // This function simulates the call and returns a descriptive message.
    console.log("Analyzing video:", videoFile.name, "with prompt:", prompt, "using gemini-2.5-pro");
    await new Promise(res => setTimeout(res, 2000));
    return `Analysis complete for video "${videoFile.name}". Gemini Pro would process this video to find key information. For example, analyzing training dynamics or identifying anomalies in a screen recording. The prompt was: "${prompt}".`;
};

const storyboardSchema = {
    type: Type.OBJECT,
    properties: {
        overallSummary: { type: Type.STRING },
        scenes: {
            type: Type.ARRAY,
            items: {
                type: Type.OBJECT,
                properties: {
                    sceneNumber: { type: Type.INTEGER },
                    startTime: { type: Type.STRING },
                    endTime: { type: Type.STRING },
                    keyframeDescription: { type: Type.STRING },
                    sceneDescription: { type: Type.STRING },
                    keyMoments: {
                        type: Type.ARRAY,
                        items: { type: Type.STRING }
                    }
                },
                required: ['sceneNumber', 'startTime', 'endTime', 'keyframeDescription', 'sceneDescription', 'keyMoments']
            }
        }
    },
    required: ['overallSummary', 'scenes']
};

export const generateStoryboardFromFile = async (videoFile: File): Promise<Storyboard> => {
    console.log(`Simulating storyboard analysis for video: ${videoFile.name}`);
    const aiInstance = getAI();

    // Step 1: Generate the storyboard structure from a text prompt.
    // This remains a simulation for file uploads, as client-side video processing is complex.
    // We make the simulation a bit smarter by using the file name for context.
    const storyboardPrompt = `
        Analyze the plausible content of a video named "${videoFile.name}" and break it down into a structured storyboard. 
        For example, if the name is "cat_playing.mp4", generate scenes of a cat playing.
        If the name is generic, assume it's about a day in the life of a software developer.

        Generate a JSON object that describes this video. It should include an overall summary and an array of distinct scenes. 
        For each scene, provide:
        - A scene number.
        - Start and end timestamps (e.g., "00:00").
        - A "keyframeDescription" which is a short, visually descriptive prompt suitable for an image generation model.
        - A "sceneDescription" summarizing the scene's events.
        - An array of "keyMoments" listing specific actions.

        The final output must be a single JSON object matching the provided schema.
    `;

    const response = await aiInstance.models.generateContent({
        model: 'gemini-2.5-pro',
        contents: storyboardPrompt,
        config: {
            responseMimeType: 'application/json',
            responseSchema: storyboardSchema
        }
    });

    const storyboardData: Storyboard = JSON.parse(response.text);

    // Step 2: Generate a keyframe image for each scene.
    const scenesWithImages = await Promise.all(storyboardData.scenes.map(async (scene) => {
        try {
            const imagePrompt = `Photorealistic, cinematic shot of: ${scene.keyframeDescription}. 16:9 aspect ratio.`;
            const imageUrl = await generateImage(imagePrompt, '16:9');
            return { ...scene, keyframeImageUrl: imageUrl };
        } catch (e) {
            console.error(`Failed to generate image for scene ${scene.sceneNumber}:`, e);
            return scene; // Return scene without image on failure
        }
    }));

    return {
        ...storyboardData,
        scenes: scenesWithImages
    };
};

export const generateStoryboardFromUrl = async (url: string): Promise<Storyboard> => {
    const aiInstance = getAI();

    const storyboardPrompt = `
      You are an expert video analyst with the ability to understand the content of YouTube videos.
      Based on the publicly available information, transcript, and visual description of the video at the URL: ${url}

      Generate a JSON object that provides a detailed storyboard. It must include an overall summary and an array of distinct scenes. 
      For each scene, provide:
      - A scene number.
      - Start and end timestamps (e.g., "00:15").
      - A "keyframeDescription" which is a short, visually descriptive prompt suitable for an image generation model to create a representative still image.
      - A "sceneDescription" summarizing what happens in the scene.
      - An array of "keyMoments" listing the most important specific actions or visual elements.

      The final output must be a single, valid JSON object matching the provided schema. Do not include any other text or markdown formatting.
    `;

    const response = await aiInstance.models.generateContent({
        model: 'gemini-2.5-pro',
        contents: storyboardPrompt,
        config: {
            responseMimeType: 'application/json',
            responseSchema: storyboardSchema
        }
    });

    const storyboardData: Storyboard = JSON.parse(response.text);

    // Step 2: Generate a keyframe image for each scene.
    const scenesWithImages = await Promise.all(storyboardData.scenes.map(async (scene) => {
        try {
            const imagePrompt = `Photorealistic, cinematic shot of: ${scene.keyframeDescription}. 16:9 aspect ratio.`;
            const imageUrl = await generateImage(imagePrompt, '16:9');
            return { ...scene, keyframeImageUrl: imageUrl };
        } catch (e) {
            console.error(`Failed to generate image for scene ${scene.sceneNumber}:`, e);
            return scene;
        }
    }));

    return {
        ...storyboardData,
        scenes: scenesWithImages
    };
};

// --- Image Generation ---

export const generateImage = async (prompt: string, aspectRatio: string): Promise<string> => {
    const aiInstance = getAI();
    const response = await aiInstance.models.generateImages({
        model: 'imagen-4.0-generate-001',
        prompt,
        config: {
            numberOfImages: 1,
            outputMimeType: 'image/jpeg',
            aspectRatio,
        },
    });
    const base64ImageBytes: string = response.generatedImages[0].image.imageBytes;
    return `data:image/jpeg;base64,${base64ImageBytes}`;
};

// --- Image Editing ---

export const editImage = async (prompt: string, imageBase64: string, mimeType: string): Promise<string> => {
    const aiInstance = getAI();
    const imagePart = { inlineData: { data: imageBase64, mimeType } };
    const textPart = { text: prompt };

    const response = await aiInstance.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts: [imagePart, textPart] },
        config: {
            responseModalities: [Modality.IMAGE],
        },
    });
    const part = response.candidates?.[0]?.content?.parts[0];
    if (part?.inlineData) {
        return `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
    }
    throw new Error("No image generated from edit.");
};


// --- Video Generation ---

export const generateVideoFromText = async (prompt: string, aspectRatio: '16:9' | '9:16') => {
    const aiInstance = getVeoAI();
    return await aiInstance.models.generateVideos({
        model: 'veo-3.1-fast-generate-preview',
        prompt,
        config: {
            numberOfVideos: 1,
            resolution: '720p',
            aspectRatio,
        }
    });
};

export const generateVideoFromImage = async (prompt: string, imageBase64: string, mimeType: string, aspectRatio: '16:9' | '9:16') => {
    const aiInstance = getVeoAI();
    return await aiInstance.models.generateVideos({
        model: 'veo-3.1-fast-generate-preview',
        prompt,
        image: {
            imageBytes: imageBase64,
            mimeType,
        },
        config: {
            numberOfVideos: 1,
            resolution: '720p',
            aspectRatio,
        }
    });
};

export const pollVideoOperation = async (operation: any) => {
    const aiInstance = getVeoAI();
    return await aiInstance.operations.getVideosOperation({ operation: operation });
}

// --- Text to Speech ---
export const generateSpeech = async (text: string): Promise<string> => {
    const aiInstance = getAI();
    const response = await aiInstance.models.generateContent({
        model: "gemini-2.5-flash-preview-tts",
        contents: [{ parts: [{ text }] }],
        config: {
            responseModalities: [Modality.AUDIO],
            speechConfig: {
                voiceConfig: {
                    prebuiltVoiceConfig: { voiceName: 'Kore' },
                },
            },
        },
    });
    
    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (base64Audio) {
        return base64Audio;
    }
    throw new Error("No audio data received from TTS API.");
};